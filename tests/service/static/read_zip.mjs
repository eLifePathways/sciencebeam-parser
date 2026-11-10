/*
 * Reads a zip given as base64 and reports what it found.
 *
 * The browser's own decompression is what the reader uses, and node has the
 * same one, so the suite can build an archive and check what comes back out.
 * The source counts the bytes it hands over, which is how the test that the
 * reader leaves large entries alone can tell.
 */
import { readZipEntries } from '../../../sciencebeam_parser/service/static/read-zip.js';

function getCountingSource(bytes) {
  const counter = { bytesRead: 0 };
  return [{
    size: bytes.length,
    read: async (start, end) => {
      counter.bytesRead += end - start;
      return bytes.subarray(start, end);
    }
  }, counter];
}

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', async () => {
  const { zip, read } = JSON.parse(input);
  const [source, counter] = getCountingSource(Buffer.from(zip, 'base64'));
  try {
    const entries = await readZipEntries(source);
    process.stdout.write(JSON.stringify({
      entries: await Promise.all(entries.map(async (entry) => ({
        name: entry.name,
        size: entry.size,
        text: (!read || read.includes(entry.name))
          ? new TextDecoder().decode(await entry.read())
          : null
      }))),
      bytesRead: counter.bytesRead
    }));
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: error.message }));
  }
});
