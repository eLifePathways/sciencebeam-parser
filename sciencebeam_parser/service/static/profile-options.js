/*
 * Which profiles the menu offers, in what order, and under which headings.
 *
 * A profile that tracks another serves exactly what that one serves, so
 * offering both is offering one conversion twice under two names. The menu
 * keeps the tracking name, which is the one that says what it is for, and drops
 * the profile it tracks; a response still reports the profile that answered, so
 * which combination ran is not lost.
 *
 * A profile nothing tracks is a choice of its own and stays, and so does the
 * deployment's own default even where something tracks it: dropping it would
 * leave the menu with nothing selected, and the browser would then pick the
 * first option and convert under a profile the deployment did not choose. A deployment that
 * makes every profile selectable has far more of those than of the others --
 * one per model whose contribution can be measured -- so the names maintained
 * as a recommendation lead, and the headings say which to pick rather than how
 * the two differ -- what a name resolves to, and whether that moves between
 * releases, is what the description beside the menu is for. A deployment
 * serving one kind gets no heading, since a sole heading divides nothing.
 *
 * Within a heading the order is what the menu shows rather than the profile
 * names behind it, which otherwise reads at random.
 */

export const TRACKING_GROUP_LABEL = 'For most uses';
export const PINNED_GROUP_LABEL = 'For specific needs';

function getLabel(profile) {
  return (profile.label || profile.name || '').toLowerCase();
}

function byLabel(a, b) {
  const left = getLabel(a);
  const right = getLabel(b);
  if (left === right) {
    return 0;
  }
  return left < right ? -1 : 1;
}

export function getProfileOptionGroups(profiles) {
  const tracked = new Set(
    profiles.map((profile) => profile.tracks).filter((name) => name)
  );
  const offered = profiles.filter(
    (profile) => profile.is_default || !tracked.has(profile.name)
  );
  const tracking = offered.filter((profile) => profile.tracks).sort(byLabel);
  const pinned = offered.filter((profile) => !profile.tracks).sort(byLabel);
  if (!tracking.length || !pinned.length) {
    const only = tracking.length ? tracking : pinned;
    return only.length ? [{ label: null, profiles: only }] : [];
  }
  return [
    { label: TRACKING_GROUP_LABEL, profiles: tracking },
    { label: PINNED_GROUP_LABEL, profiles: pinned }
  ];
}

export function getTrackedProfile(profiles, profile) {
  if (!profile || !profile.tracks) {
    return null;
  }
  return profiles.find((item) => item.name === profile.tracks) || null;
}
