/*
 * Reads a zip given as base64 on stdin and writes what it found.
 *
 * The browser's own decompression is what the reader uses, and node has the
 * same one, so the suite can build an archive and check what comes back out.
 */
import { readZipEntries } from '../../../sciencebeam_parser/service/static/read-zip.js';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', async () => {
  try {
    const entries = await readZipEntries(Buffer.from(input.trim(), 'base64'));
    process.stdout.write(JSON.stringify({
      entries: entries.map((entry) => ({
        name: entry.name,
        text: new TextDecoder().decode(entry.data)
      }))
    }));
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: error.message }));
  }
});
