/*
 * Which profiles the menu offers.
 *
 * A profile that tracks another serves exactly what that one serves, so
 * offering both is offering one conversion twice under two names. The menu
 * keeps the tracking name, which is the one that says what it is for, and drops
 * the profile it tracks; a response still reports the profile that answered, so
 * which combination ran is not lost.
 *
 * A profile nothing tracks is a choice of its own and stays.
 */

export function getProfileOptions(profiles) {
  const tracked = new Set(
    profiles.map((profile) => profile.tracks).filter((name) => name)
  );
  return profiles.filter((profile) => !tracked.has(profile.name));
}

export function getTrackedProfile(profiles, profile) {
  if (!profile || !profile.tracks) {
    return null;
  }
  return profiles.find((item) => item.name === profile.tracks) || null;
}
