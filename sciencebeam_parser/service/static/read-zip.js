/*
 * Reading the entries of a zip, with nothing but what the browser has.
 *
 * The service answers with an archive when asked for one, and what is in it --
 * the XML and the figures the document carries -- is the part worth looking
 * at. Shipping an archive library to open one would cost more than this does.
 *
 * It reads what this service writes and what an ordinary zip is: stored or
 * deflated entries, no encryption, no zip64. Anything else is refused by name
 * rather than read wrongly. It takes an `ArrayBuffer` or any view of one,
 * because a view onto a larger buffer is what most ways of getting bytes hand
 * back, and reading it as though it started at zero finds no archive at all.
 *
 * Each entry states its size and is read when asked rather than up front. One
 * page of vector drawing can be tens of megabytes on its own, and a caller
 * that is going to leave an entry alone should not pay to decompress it.
 */

const END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x06054b50;
const CENTRAL_DIRECTORY_SIGNATURE = 0x02014b50;
const LOCAL_FILE_SIGNATURE = 0x04034b50;

const END_OF_CENTRAL_DIRECTORY_LENGTH = 22;
const MAX_COMMENT_LENGTH = 0xffff;

const STORED = 0;
const DEFLATED = 8;

const ZIP64_MARKER_16 = 0xffff;
const ZIP64_MARKER_32 = 0xffffffff;

function getEndOfCentralDirectoryOffset(view) {
  const earliest = Math.max(
    0, view.byteLength - END_OF_CENTRAL_DIRECTORY_LENGTH - MAX_COMMENT_LENGTH
  );
  for (
    let offset = view.byteLength - END_OF_CENTRAL_DIRECTORY_LENGTH;
    offset >= earliest;
    offset -= 1
  ) {
    if (view.getUint32(offset, true) === END_OF_CENTRAL_DIRECTORY_SIGNATURE) {
      return offset;
    }
  }
  throw new Error('not a zip: no end of central directory');
}

async function inflateRaw(data) {
  const stream = new Blob([data]).stream().pipeThrough(
    new DecompressionStream('deflate-raw')
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function getEntryData(bytes, view, entry) {
  if (view.getUint32(entry.localOffset, true) !== LOCAL_FILE_SIGNATURE) {
    throw new Error(`not a zip: ${entry.name} has no local header`);
  }
  const start = (
    entry.localOffset
    + 30
    + view.getUint16(entry.localOffset + 26, true)
    + view.getUint16(entry.localOffset + 28, true)
  );
  const data = bytes.subarray(start, start + entry.compressedSize);
  if (entry.method === STORED) {
    return data;
  }
  if (entry.method === DEFLATED) {
    return inflateRaw(data);
  }
  throw new Error(`${entry.name} uses compression method ${entry.method}`);
}

function getCentralDirectoryEntries(view, decoder, start, count) {
  const entries = [];
  let offset = start;
  for (let index = 0; index < count; index += 1) {
    if (view.getUint32(offset, true) !== CENTRAL_DIRECTORY_SIGNATURE) {
      throw new Error('not a zip: the central directory ends early');
    }
    const flags = view.getUint16(offset + 8, true);
    const nameLength = view.getUint16(offset + 28, true);
    const name = decoder.decode(
      new Uint8Array(view.buffer, view.byteOffset + offset + 46, nameLength)
    );
    if (flags & 1) {
      throw new Error(`${name} is encrypted`);
    }
    const compressedSize = view.getUint32(offset + 20, true);
    const size = view.getUint32(offset + 24, true);
    const localOffset = view.getUint32(offset + 42, true);
    if (
      compressedSize === ZIP64_MARKER_32
      || size === ZIP64_MARKER_32
      || localOffset === ZIP64_MARKER_32
    ) {
      throw new Error(`${name} needs zip64, which this does not read`);
    }
    if (!name.endsWith('/')) {
      entries.push({
        name,
        size,
        method: view.getUint16(offset + 10, true),
        compressedSize,
        localOffset
      });
    }
    offset += (
      46
      + nameLength
      + view.getUint16(offset + 30, true)
      + view.getUint16(offset + 32, true)
    );
  }
  return entries;
}

export function readZipEntries(buffer) {
  const bytes = ArrayBuffer.isView(buffer)
    ? new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
    : new Uint8Array(buffer);
  const view = new DataView(
    bytes.buffer, bytes.byteOffset, bytes.byteLength
  );
  const endOffset = getEndOfCentralDirectoryOffset(view);
  const count = view.getUint16(endOffset + 10, true);
  const directoryOffset = view.getUint32(endOffset + 16, true);
  if (count === ZIP64_MARKER_16 || directoryOffset === ZIP64_MARKER_32) {
    throw new Error('the archive needs zip64, which this does not read');
  }
  const decoder = new TextDecoder();
  return getCentralDirectoryEntries(view, decoder, directoryOffset, count).map(
    (entry) => ({
      name: entry.name,
      size: entry.size,
      read: () => getEntryData(bytes, view, entry)
    })
  );
}
