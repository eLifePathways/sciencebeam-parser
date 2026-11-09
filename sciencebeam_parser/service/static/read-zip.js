/*
 * Reading the entries of a zip, with nothing but what the browser has.
 *
 * The service answers with an archive when asked for one, and what is in it --
 * the XML and the figures the document carries -- is the part worth looking
 * at. Shipping an archive library to open one would cost more than this does.
 *
 * It reads what this service writes and what an ordinary zip is: stored or
 * deflated entries, no encryption, no zip64. Anything else is refused by name
 * rather than read wrongly.
 *
 * It reads through a source that can hand back a range of bytes, rather than
 * from the bytes themselves, and each entry states its size and is read only
 * when asked for. A real document in this corpus answers with a 74MB archive
 * whose largest entry is one page of vector drawing at 51MB; holding all of
 * that in memory to read the 124KB of XML beside it is what makes a tab give
 * up. `getZipSourceForBlob` keeps it where the response already is.
 */

const END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x06054b50;
const CENTRAL_DIRECTORY_SIGNATURE = 0x02014b50;
const LOCAL_FILE_SIGNATURE = 0x04034b50;

const END_OF_CENTRAL_DIRECTORY_LENGTH = 22;
const MAX_COMMENT_LENGTH = 0xffff;
const LOCAL_FILE_HEADER_LENGTH = 30;
const CENTRAL_DIRECTORY_ENTRY_LENGTH = 46;

const STORED = 0;
const DEFLATED = 8;

const ZIP64_MARKER_16 = 0xffff;
const ZIP64_MARKER_32 = 0xffffffff;

export function getZipSourceForBytes(buffer) {
  const bytes = ArrayBuffer.isView(buffer)
    ? new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
    : new Uint8Array(buffer);
  return {
    size: bytes.length,
    read: async (start, end) => bytes.subarray(start, end)
  };
}

export function getZipSourceForBlob(blob) {
  return {
    size: blob.size,
    read: async (start, end) => new Uint8Array(
      await blob.slice(start, end).arrayBuffer()
    )
  };
}

function getView(bytes) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function getEndOfCentralDirectoryOffset(tailView) {
  for (
    let offset = tailView.byteLength - END_OF_CENTRAL_DIRECTORY_LENGTH;
    offset >= 0;
    offset -= 1
  ) {
    if (tailView.getUint32(offset, true) === END_OF_CENTRAL_DIRECTORY_SIGNATURE) {
      return offset;
    }
  }
  throw new Error('not a zip: no end of central directory');
}

async function getEndOfCentralDirectory(source) {
  const tailLength = Math.min(
    source.size, END_OF_CENTRAL_DIRECTORY_LENGTH + MAX_COMMENT_LENGTH
  );
  const tailView = getView(
    await source.read(source.size - tailLength, source.size)
  );
  const offset = getEndOfCentralDirectoryOffset(tailView);
  const count = tailView.getUint16(offset + 10, true);
  const directoryOffset = tailView.getUint32(offset + 16, true);
  if (count === ZIP64_MARKER_16 || directoryOffset === ZIP64_MARKER_32) {
    throw new Error('the archive needs zip64, which this does not read');
  }
  return {
    count,
    directoryOffset,
    directorySize: tailView.getUint32(offset + 12, true)
  };
}

async function inflateRaw(data) {
  const stream = new Blob([data]).stream().pipeThrough(
    new DecompressionStream('deflate-raw')
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function getEntryData(source, entry) {
  const headerView = getView(await source.read(
    entry.localOffset, entry.localOffset + LOCAL_FILE_HEADER_LENGTH
  ));
  if (headerView.getUint32(0, true) !== LOCAL_FILE_SIGNATURE) {
    throw new Error(`not a zip: ${entry.name} has no local header`);
  }
  const start = (
    entry.localOffset
    + LOCAL_FILE_HEADER_LENGTH
    + headerView.getUint16(26, true)
    + headerView.getUint16(28, true)
  );
  const data = await source.read(start, start + entry.compressedSize);
  if (entry.method === STORED) {
    return data;
  }
  if (entry.method === DEFLATED) {
    return inflateRaw(data);
  }
  throw new Error(`${entry.name} uses compression method ${entry.method}`);
}

function getCentralDirectoryEntries(directoryBytes, count) {
  const view = getView(directoryBytes);
  const decoder = new TextDecoder();
  const entries = [];
  let offset = 0;
  for (let index = 0; index < count; index += 1) {
    if (
      offset + CENTRAL_DIRECTORY_ENTRY_LENGTH > view.byteLength
      || view.getUint32(offset, true) !== CENTRAL_DIRECTORY_SIGNATURE
    ) {
      throw new Error('not a zip: the central directory ends early');
    }
    const flags = view.getUint16(offset + 8, true);
    const nameLength = view.getUint16(offset + 28, true);
    const name = decoder.decode(directoryBytes.subarray(
      offset + CENTRAL_DIRECTORY_ENTRY_LENGTH,
      offset + CENTRAL_DIRECTORY_ENTRY_LENGTH + nameLength
    ));
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
        name, size, compressedSize, localOffset,
        method: view.getUint16(offset + 10, true)
      });
    }
    offset += (
      CENTRAL_DIRECTORY_ENTRY_LENGTH
      + nameLength
      + view.getUint16(offset + 30, true)
      + view.getUint16(offset + 32, true)
    );
  }
  return entries;
}

export async function readZipEntries(source) {
  const { count, directoryOffset, directorySize } = (
    await getEndOfCentralDirectory(source)
  );
  const directoryBytes = await source.read(
    directoryOffset, directoryOffset + directorySize
  );
  return getCentralDirectoryEntries(directoryBytes, count).map((entry) => ({
    name: entry.name,
    size: entry.size,
    read: () => getEntryData(source, entry)
  }));
}
