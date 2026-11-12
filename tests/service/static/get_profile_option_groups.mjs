/*
 * Runs the profile menu's grouping over a profile list on stdin.
 */
import {
  getProfileOptionGroups
} from '../../../sciencebeam_parser/service/static/profile-options.js';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  process.stdout.write(JSON.stringify(getProfileOptionGroups(JSON.parse(input))));
});
