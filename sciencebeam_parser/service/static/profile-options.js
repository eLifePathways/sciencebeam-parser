/*
 * How the profile menu is divided.
 *
 * A profile that tracks another serves that one's models, so a flat list would
 * offer the same combination twice under two names. The groups separate the two
 * choices: follow what the service recommends, or stay on one combination.
 */

export const TRACKING_GROUP_LABEL = 'Recommended';
export const PINNED_GROUP_LABEL = 'A specific combination';

export function getProfileOptionGroups(profiles) {
  const tracking = profiles.filter((profile) => profile.tracks);
  const pinned = profiles.filter((profile) => !profile.tracks);
  if (!tracking.length || !pinned.length) {
    // Nothing to tell apart, and a sole group heading is noise.
    return profiles.length ? [{ label: null, profiles: profiles }] : [];
  }
  return [
    { label: TRACKING_GROUP_LABEL, profiles: tracking },
    { label: PINNED_GROUP_LABEL, profiles: pinned }
  ];
}
