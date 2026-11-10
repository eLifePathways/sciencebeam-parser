/* Applies the view box to the SVG given on stdin. */
import { addSvgViewBox } from '../../../sciencebeam_parser/service/static/svg-view-box.js';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  process.stdout.write(addSvgViewBox(input));
});
