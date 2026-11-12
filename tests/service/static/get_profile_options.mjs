/*
 * Runs the profile menu's choice, order and headings over a profile list on stdin.
 */
import {
  getProfileOptionGroups,
  getTrackedProfile
} from '../../../sciencebeam_parser/service/static/profile-options.js';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const profiles = JSON.parse(input);
  process.stdout.write(JSON.stringify({
    groups: getProfileOptionGroups(profiles).map((group) => ({
      label: group.label,
      profiles: group.profiles.map((profile) => profile.name)
    })),
    tracked: Object.fromEntries(profiles.map((profile) => [
      profile.name,
      (getTrackedProfile(profiles, profile) || {}).name || null
    ]))
  }));
});
