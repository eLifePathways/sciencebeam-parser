/*
 * Runs the profile menu's choice of options over a profile list on stdin.
 */
import {
  getProfileOptions,
  getTrackedProfile
} from '../../../sciencebeam_parser/service/static/profile-options.js';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const profiles = JSON.parse(input);
  process.stdout.write(JSON.stringify({
    options: getProfileOptions(profiles).map((profile) => profile.name),
    tracked: Object.fromEntries(profiles.map((profile) => [
      profile.name,
      (getTrackedProfile(profiles, profile) || {}).name || null
    ]))
  }));
});
