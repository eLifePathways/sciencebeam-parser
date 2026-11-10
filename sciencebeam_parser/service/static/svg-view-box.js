/*
 * Giving a drawing the coordinate system it never declared.
 *
 * The SVGs extracted from a PDF carry no width, height or viewBox, so a
 * browser draws them into the default 300x150 box and clips whatever the
 * document put outside it. A viewBox is what lets the same drawing scale to
 * whatever space it is given instead.
 *
 * The extent comes from the coordinates in the path data. One range is used
 * for both axes rather than a range each: the commands differ in how many
 * numbers they carry, so which number is an x and which is a y is not
 * something this can tell, and a square that covers both never clips what a
 * pair of narrower ones might.
 */

const SVG_ROOT_PATTERN = /<svg\b[^>]*>/;
const PATH_DATA_PATTERN = /\sd="([^"]*)"/g;
const NUMBER_PATTERN = /-?\d+(?:\.\d+)?/g;

function getCoordinateRange(svgText) {
  let lowest = Infinity;
  let highest = -Infinity;
  let pathMatch = PATH_DATA_PATTERN.exec(svgText);
  while (pathMatch) {
    const numbers = pathMatch[1].match(NUMBER_PATTERN);
    (numbers || []).forEach((text) => {
      const value = Number(text);
      lowest = Math.min(lowest, value);
      highest = Math.max(highest, value);
    });
    pathMatch = PATH_DATA_PATTERN.exec(svgText);
  }
  PATH_DATA_PATTERN.lastIndex = 0;
  return { lowest, highest };
}

export function addSvgViewBox(svgText) {
  const rootMatch = SVG_ROOT_PATTERN.exec(svgText);
  if (!rootMatch) {
    return svgText;
  }
  const root = rootMatch[0];
  if (/\bviewBox=/.test(root) || (/\bwidth=/.test(root) && /\bheight=/.test(root))) {
    return svgText;
  }
  const { lowest, highest } = getCoordinateRange(svgText);
  if (!Number.isFinite(lowest) || highest <= lowest) {
    return svgText;
  }
  const round = (value) => Number(value.toFixed(3));
  const viewBox = (
    ` viewBox="${round(lowest)} ${round(lowest)}`
    + ` ${round(highest - lowest)} ${round(highest - lowest)}"`
  );
  return (
    svgText.slice(0, rootMatch.index)
    + root.replace(/\s*\/?>$/, (end) => viewBox + end)
    + svgText.slice(rootMatch.index + root.length)
  );
}
