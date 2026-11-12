import {
  getGraphicHrefs,
  getGraphicReferences,
  renderArticleViewModel
} from './article-view.js';
import { parseJatsViewModel } from './jats-to-html.js';
import { parseTeiViewModel } from './tei-to-html.js';
import { getZipSourceForBlob, readZipEntries } from './read-zip.js';
import { addSvgViewBox } from './svg-view-box.js';
import { getProfileOptions, getTrackedProfile } from './profile-options.js';


const deployment = JSON.parse(document.getElementById('deployment').textContent);

const PROFILE_HEADER = 'x-sciencebeam-profile';
const PROFILE_DIGEST_HEADER = 'x-sciencebeam-profile-digest';
const PROFILE_REQUESTED_HEADER = 'x-sciencebeam-profile-requested';
const LLM_USAGE_HEADER = 'x-sciencebeam-llm-usage';
const SLOW_REQUEST_SECONDS = 10;
const PDF_MEDIA_TYPE = 'application/pdf';
const LABELS_SETTING = 'sciencebeam-show-labels';

const PARSE_BY_READING = {
  jats: parseJatsViewModel,
  tei: parseTeiViewModel
};

// What a reading draws without being asked. One page of vector drawing can be
// tens of megabytes; opening one from the file list is a click of its own.
const MAX_ASSET_BYTES = 8 * 1024 * 1024;

const IMAGE_MEDIA_TYPE_BY_EXTENSION = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp'
};

const form = document.getElementById('convert-form');
const sourceFileInput = document.getElementById('source-file');
const fileErrorElement = document.getElementById('file-error');
const profileSelect = document.getElementById('profile');
const profileHint = document.getElementById('profile-hint');
const outputSelect = document.getElementById('output');
const outputHint = document.getElementById('output-hint');
const firstPageInput = document.getElementById('first-page');
const lastPageInput = document.getElementById('last-page');
const includesElement = document.getElementById('includes');
const convertButton = document.getElementById('convert');
const stopButton = document.getElementById('stop');
const statusElement = document.getElementById('status');
const resultSection = document.getElementById('result');
const failureSection = document.getElementById('failure');
const downloadButton = document.getElementById('download');
const readingButton = document.getElementById('show-reading');
const rawButton = document.getElementById('show-raw');
const readingElement = document.getElementById('result-reading');
const resultBody = document.getElementById('result-body');
const resultNote = document.getElementById('result-note');
const downloadXmlButton = document.getElementById('download-xml');
const archiveSection = document.getElementById('archive');
const archiveSummary = document.getElementById('archive-summary');
const archiveFiles = document.getElementById('archive-files');
const resultSummary = document.getElementById('result-summary');
const archiveShowAll = document.getElementById('archive-show-all');
const archiveHideAll = document.getElementById('archive-hide-all');
const labelsControl = document.getElementById('labels-control');
const labelsCheckbox = document.getElementById('show-labels');
const sourceSection = document.getElementById('source');
const sourceNote = document.getElementById('source-note');
const sourceActions = document.getElementById('source-actions');
const showSourceButton = document.getElementById('show-source');
const sourceFrame = document.getElementById('source-frame');

let downloadUrl = null;
let sourceUrl = null;
let assetUrls = [];
let requestController = null;

function describeAcceptedMediaTypes() {
  return deployment.accepted_source_media_types
    .map((accepted) => accepted.extension)
    .join(', ');
}

function setUpSourceFileInput() {
  sourceFileInput.accept = deployment.accepted_source_media_types
    .flatMap((accepted) => [accepted.extension, accepted.media_type])
    .join(',');
  document.getElementById('accepted-hint').textContent = (
    'This service accepts ' + describeAcceptedMediaTypes() + '.'
  );
}

function isAcceptedFile(file) {
  const filename = file.name.toLowerCase();
  return deployment.accepted_source_media_types.some(
    (accepted) => (
      filename.endsWith(accepted.extension)
      || (!!file.type && file.type === accepted.media_type)
    )
  );
}

function getSelectedFile() {
  return sourceFileInput.files.length ? sourceFileInput.files[0] : null;
}

function checkSelectedFile() {
  const file = getSelectedFile();
  const rejected = !!file && !isAcceptedFile(file);
  fileErrorElement.textContent = rejected
    ? (
      file.name + ' is not a document this service can read. '
      + 'It accepts ' + describeAcceptedMediaTypes() + '.'
    )
    : '';
  fileErrorElement.hidden = !rejected;
  convertButton.disabled = rejected;
}

function addOption(select, value, label) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  select.appendChild(option);
  return option;
}

function setUpOutputSelect() {
  deployment.output_media_types.forEach((output) => {
    addOption(outputSelect, output.media_type, output.label);
  });
  outputSelect.addEventListener('change', showSelectedOutputHint);
  showSelectedOutputHint();
}

function getSelectedOutput() {
  return deployment.output_media_types.find(
    (output) => output.media_type === outputSelect.value
  );
}

function getOutputHint(output) {
  if (!output || output.reading) {
    return '';
  }
  return 'Returned as a file to download rather than as text to read here.';
}

function showSelectedOutputHint() {
  outputHint.textContent = getOutputHint(getSelectedOutput());
}

function setUpIncludes() {
  deployment.includes_groups.forEach((group) => {
    const label = document.createElement('label');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = group.field_names.join(',');
    checkbox.title = group.field_names.join(', ');
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(' ' + group.label));
    includesElement.appendChild(label);
  });
}

function getSelectedIncludes() {
  return Array.from(includesElement.querySelectorAll('input:checked'))
    .flatMap((checkbox) => checkbox.value.split(','));
}

function addProfileOptions(profiles) {
  getProfileOptions(profiles).forEach((profile) => {
    const option = addOption(profileSelect, profile.name, profile.label || profile.name);
    option.selected = profile.is_default;
  });
}

function getProfileHint(profiles, profile) {
  if (!profile) {
    return '';
  }
  const tracked = getTrackedProfile(profiles, profile);
  // Which combination it is today, since the response will name that one rather
  // than the name picked here.
  const serves = tracked
    ? 'Currently serves ' + (tracked.label || tracked.name) + '.'
    : '';
  return [profile.description || '', serves].filter((part) => part).join(' ');
}

function showProfileDescription(profiles) {
  const profile = profiles.find((item) => item.name === profileSelect.value);
  profileHint.textContent = getProfileHint(profiles, profile);
}

async function loadProfiles() {
  let profiles;
  try {
    const response = await fetch(deployment.api_path + '/profiles');
    if (!response.ok) {
      throw new Error('responded with ' + response.status);
    }
    profiles = (await response.json()).profiles;
  } catch (error) {
    profileHint.textContent = (
      'The profiles this service offers could not be listed (' + error.message
      + '). The request will be served by its default profile.'
    );
    profileSelect.disabled = true;
    addOption(profileSelect, '', 'default');
    return;
  }
  addProfileOptions(profiles);
  profileSelect.addEventListener('change', () => showProfileDescription(profiles));
  showProfileDescription(profiles);
}

function getRequestParams() {
  const params = new URLSearchParams();
  if (profileSelect.value) {
    params.set('profile', profileSelect.value);
  }
  if (firstPageInput.value) {
    params.set('first_page', firstPageInput.value);
  }
  if (lastPageInput.value) {
    params.set('last_page', lastPageInput.value);
  }
  const includes = getSelectedIncludes();
  if (includes.length) {
    params.set('includes', includes.join(','));
  }
  return params.toString();
}

function getRequestUrl() {
  const params = getRequestParams();
  return deployment.api_path + '/convert' + (params ? '?' + params : '');
}

function quoteForShell(value) {
  return "'" + value.replace(/'/g, `'\\''`) + "'";
}

function getEquivalentCurlCommand(file) {
  return [
    'curl --fail-with-body -X POST',
    '  ' + quoteForShell(new URL(getRequestUrl(), window.location.href).href),
    '  -H ' + quoteForShell('Accept: ' + outputSelect.value),
    '  -F ' + quoteForShell('input=@' + file.name)
  ].join(' \\\n');
}

function describeBytes(count) {
  if (count >= 1024 * 1024) {
    return (count / (1024 * 1024)).toFixed(1) + 'MB';
  }
  if (count >= 1024) {
    return Math.round(count / 1024) + 'KB';
  }
  return count + ' bytes';
}

/* The body, collected here rather than by the browser's own blob machinery.
 *
 * `response.blob()` has the browser stream the body into blob storage itself,
 * and that is the one step that fails for a large response where every part
 * of it separately does not: the same bytes read as a stream arrive in full,
 * and a blob built in the page holds far more than any response here. So the
 * chunks are read and then made into a blob, which is those two steps and not
 * the one in between.
 *
 * Counting them also says how far a body got when it does stop, which is what
 * tells a response that ended early from a request that never arrived. */
async function getResponseBlob(response) {
  const expected = Number(response.headers.get('content-length')) || 0;
  if (!response.body) {
    return response.blob();
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  let reported = -1;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(value);
      received += value.byteLength;
      const megabytes = Math.floor(received / (1024 * 1024));
      if (megabytes !== reported) {
        reported = megabytes;
        statusElement.textContent = (
          'Receiving ' + describeBytes(received)
          + (expected ? ' of ' + describeBytes(expected) : '') + '.'
        );
      }
    }
  } catch (error) {
    throw new Error(
      'the body stopped after ' + describeBytes(received)
      + (expected ? ' of ' + describeBytes(expected) : '')
      + ', while the status was ' + response.status
      + ' (' + error.message + ')'
    );
  }
  return new Blob(chunks, {
    type: response.headers.get('content-type') || ''
  });
}

function addMetaRow(listElement, name, value) {
  if (!value) {
    return;
  }
  const term = document.createElement('dt');
  term.textContent = name;
  const description = document.createElement('dd');
  description.textContent = value;
  listElement.appendChild(term);
  listElement.appendChild(description);
}

function getDownloadFilename(file) {
  const output = getSelectedOutput();
  const baseName = file.name.replace(/\.[^.]+$/, '');
  return baseName + (output ? output.extension : '');
}

function releaseDownloadUrl() {
  if (downloadUrl) {
    URL.revokeObjectURL(downloadUrl);
    downloadUrl = null;
  }
}

function releaseAssetUrls() {
  assetUrls.forEach((url) => URL.revokeObjectURL(url));
  assetUrls = [];
}

function getImageMediaType(name) {
  const lowerName = name.toLowerCase();
  const extension = Object.keys(IMAGE_MEDIA_TYPE_BY_EXTENSION).find(
    (candidate) => lowerName.endsWith(candidate)
  );
  return extension ? IMAGE_MEDIA_TYPE_BY_EXTENSION[extension] : null;
}

async function getAssetUrlByName(entries, wantedNames) {
  const urlByName = {};
  const tooLarge = [];
  const wantedEntries = entries.filter(
    (entry) => wantedNames.includes(entry.name) && getImageMediaType(entry.name)
  );
  await Promise.all(wantedEntries.map(async (entry) => {
    if (entry.size > MAX_ASSET_BYTES) {
      tooLarge.push(entry.name);
      return;
    }
    const url = URL.createObjectURL(await getImageBlob(entry));
    assetUrls.push(url);
    urlByName[entry.name] = url;
  }));
  return { urlByName, tooLarge };
}

function releaseSourceUrl() {
  if (sourceUrl) {
    URL.revokeObjectURL(sourceUrl);
    sourceUrl = null;
  }
}

/* The viewer loads when the pane is opened, not before.
 *
 * Handing a PDF to the browser's viewer costs something, and the pane starts
 * closed because most of the time the result is what is being looked at. */
function loadSourceFrame() {
  if (!sourceUrl || sourceFrame.getAttribute('src') === sourceUrl) {
    return;
  }
  sourceFrame.src = sourceUrl;
}

function showSourceBlob(blob) {
  releaseSourceUrl();
  sourceUrl = URL.createObjectURL(blob);
  sourceFrame.hidden = false;
  if (sourceSection.open) {
    loadSourceFrame();
  }
}

function isPdfFile(file) {
  return file.name.toLowerCase().endsWith('.pdf') || file.type === PDF_MEDIA_TYPE;
}

async function loadConvertedSource(file) {
  showSourceButton.disabled = true;
  sourceNote.textContent = 'Converting it to the PDF the parser reads.';
  const formData = new FormData();
  formData.append('input', file);
  try {
    const response = await fetch(deployment.api_path + '/convert', {
      method: 'POST',
      headers: { Accept: PDF_MEDIA_TYPE },
      body: formData
    });
    if (!response.ok) {
      sourceNote.textContent = (
        'The PDF the parser reads could not be shown: ' + response.status + ' '
        + response.statusText + '.'
      );
      return;
    }
    sourceNote.textContent = (
      'The PDF the parser read, converted from ' + file.name + '.'
    );
    sourceActions.hidden = true;
    showSourceBlob(await response.blob());
  } catch (error) {
    sourceNote.textContent = (
      'The PDF the parser reads could not be fetched: ' + error.message
    );
  } finally {
    showSourceButton.disabled = false;
  }
}

function showSource(file) {
  releaseSourceUrl();
  sourceFrame.hidden = true;
  sourceFrame.removeAttribute('src');
  if (isPdfFile(file)) {
    sourceActions.hidden = true;
    sourceNote.textContent = file.name;
    showSourceBlob(
      file.type ? file : new Blob([file], { type: PDF_MEDIA_TYPE })
    );
  } else {
    sourceActions.hidden = false;
    sourceNote.textContent = (
      'The parser reads a PDF converted from ' + file.name + '. '
      + 'Asking for it is a second request.'
    );
    showSourceButton.onclick = () => loadConvertedSource(file);
  }
  sourceSection.hidden = false;
}

function setUpDownload(blob, filename) {
  releaseDownloadUrl();
  downloadUrl = URL.createObjectURL(blob);
  downloadButton.onclick = () => {
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    link.click();
  };
}

function showReading(isReading) {
  readingElement.hidden = !isReading;
  resultBody.hidden = isReading;
  // Which one is showing, rather than which one cannot be pressed: a disabled
  // button reads as unavailable, not as chosen.
  readingButton.classList.toggle('selected', isReading);
  rawButton.classList.toggle('selected', !isReading);
  readingButton.setAttribute('aria-pressed', String(isReading));
  rawButton.setAttribute('aria-pressed', String(!isReading));
  labelsControl.hidden = !isReading;
}

function applyLabels() {
  readingElement.classList.toggle('jats-labelled', labelsCheckbox.checked);
}

function setUpLabels() {
  try {
    labelsCheckbox.checked = window.localStorage.getItem(LABELS_SETTING) === 'true';
  } catch (error) {
    labelsCheckbox.checked = false;
  }
  applyLabels();
  labelsCheckbox.addEventListener('change', () => {
    applyLabels();
    try {
      window.localStorage.setItem(LABELS_SETTING, String(labelsCheckbox.checked));
    } catch (error) {
      // how one person reads is not worth failing a conversion over
    }
  });
}

async function getArchiveAssets(content, viewModel) {
  if (!content.entries) {
    return {};
  }
  const { urlByName, tooLarge } = await getAssetUrlByName(
    content.entries, getGraphicHrefs(viewModel)
  );
  if (tooLarge.length) {
    resultNote.textContent += (
      ' Not shown, over ' + Math.round(MAX_ASSET_BYTES / (1024 * 1024))
      + 'MB each: ' + tooLarge.join(', ') + '.'
    );
  }
  return urlByName;
}

async function showReadableResult(content, output) {
  readingElement.replaceChildren();
  let viewModel;
  try {
    viewModel = PARSE_BY_READING[output.reading](content.text);
  } catch (error) {
    resultNote.textContent = (
      'It could not be read as ' + output.reading.toUpperCase() + ': '
      + error.message
    );
    readingButton.hidden = true;
    rawButton.hidden = true;
    showReading(false);
    return null;
  }
  const urlByName = await getArchiveAssets(content, viewModel);
  readingElement.appendChild(renderArticleViewModel(viewModel, document, {
    getAssetUrl: (href) => urlByName[href] || null
  }));
  readingButton.hidden = false;
  rawButton.hidden = false;
  showReading(true);
  return viewModel;
}

function showUnreadableResult(output, keepNote) {
  readingElement.replaceChildren();
  readingButton.hidden = true;
  rawButton.hidden = true;
  if (!keepNote) {
    resultNote.textContent = getOutputHint(output);
  }
  showReading(false);
}

async function getArchiveContentOrNote(blob, output) {
  try {
    return await getArchiveContent(blob, output);
  } catch (error) {
    resultNote.textContent = 'The archive could not be read: ' + error.message;
    return null;
  }
}

async function getArchiveContent(blob, output) {
  const entries = await readZipEntries(getZipSourceForBlob(blob));
  const xmlEntry = entries.find((entry) => entry.name === output.archive_xml_name);
  if (!xmlEntry) {
    throw new Error(
      'the archive carries no ' + output.archive_xml_name
    );
  }
  return {
    text: new TextDecoder().decode(await xmlEntry.read()),
    entries
  };
}

function addCell(row, text, className) {
  const cell = document.createElement('td');
  if (className) {
    cell.className = className;
  }
  cell.textContent = text;
  row.appendChild(cell);
  return cell;
}

function appendTo(parent, tagName, className) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  parent.appendChild(element);
  return element;
}

const assetUrlByEntryName = {};

async function getImageBlob(entry) {
  const mediaType = getImageMediaType(entry.name);
  const data = await entry.read();
  if (mediaType !== 'image/svg+xml') {
    return new Blob([data], { type: mediaType });
  }
  return new Blob(
    [addSvgViewBox(new TextDecoder().decode(data))], { type: mediaType }
  );
}

async function getArchiveFileUrl(entry) {
  if (!assetUrlByEntryName[entry.name]) {
    const url = URL.createObjectURL(await getImageBlob(entry));
    assetUrls.push(url);
    assetUrlByEntryName[entry.name] = url;
  }
  return assetUrlByEntryName[entry.name];
}

async function openArchiveFile(entry, previewRow, button) {
  const cell = previewRow.firstChild;
  cell.replaceChildren();
  previewRow.hidden = false;
  button.textContent = 'Hide';
  const caption = appendTo(cell, 'p', 'hint');
  caption.textContent = 'Reading ' + describeBytes(entry.size) + '.';
  const url = await getArchiveFileUrl(entry);
  caption.replaceChildren();
  const link = appendTo(caption, 'a');
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener';
  link.textContent = entry.name + ', ' + describeBytes(entry.size)
    + ' — open full size';
  const image = appendTo(cell, 'img', 'archive-image');
  image.src = url;
  image.alt = entry.name;
}

function closeArchiveFile(previewRow, button) {
  previewRow.hidden = true;
  previewRow.firstChild.replaceChildren();
  button.textContent = 'View';
}

function describeFigure(name) {
  // A figure's own label is usually just its number.
  return /^\d+$/.test(name) ? 'figure ' + name : name;
}

function addArchiveFileRow(entry, figuresByHref) {
  const row = appendTo(archiveFiles, 'tr');
  addCell(row, entry.name, 'archive-name');
  addCell(row, describeBytes(entry.size), 'archive-size');
  const figures = figuresByHref[entry.name];
  addCell(
    row,
    figures ? 'used by ' + figures.map(describeFigure).join(', ') : 'not referenced',
    figures ? 'archive-used' : 'archive-unused'
  );
  const actionCell = addCell(row, '');
  const previewRow = appendTo(archiveFiles, 'tr', 'archive-preview');
  previewRow.hidden = true;
  appendTo(previewRow, 'td').colSpan = 4;
  if (!getImageMediaType(entry.name)) {
    return null;
  }
  const button = appendTo(actionCell, 'button');
  button.type = 'button';
  button.textContent = 'View';
  button.onclick = () => (
    previewRow.hidden
      ? openArchiveFile(entry, previewRow, button)
      : closeArchiveFile(previewRow, button)
  );
  return { entry, previewRow, button };
}

function setUpArchiveShowAll(viewable) {
  archiveShowAll.hidden = !viewable.length;
  archiveHideAll.hidden = !viewable.length;
  archiveShowAll.onclick = () => viewable.forEach(({ entry, previewRow, button }) => {
    if (previewRow.hidden) {
      openArchiveFile(entry, previewRow, button);
    }
  });
  archiveHideAll.onclick = () => viewable.forEach(
    ({ previewRow, button }) => closeArchiveFile(previewRow, button)
  );
}

function showArchiveFiles(content, viewModel, output) {
  archiveFiles.replaceChildren();
  if (!content || !content.entries) {
    archiveSection.hidden = true;
    return;
  }
  const figuresByHref = {};
  if (viewModel) {
    getGraphicReferences(viewModel).forEach(({ href, figure }) => {
      figuresByHref[href] = (figuresByHref[href] || []).concat(figure);
    });
  }
  const assetEntries = content.entries.filter(
    (entry) => entry.name !== output.archive_xml_name
  );
  setUpArchiveShowAll(
    assetEntries
      .map((entry) => addArchiveFileRow(entry, figuresByHref))
      .filter((viewable) => viewable)
  );
  const unreferenced = assetEntries.filter(
    (entry) => !figuresByHref[entry.name]
  ).length;
  archiveSummary.textContent = (
    assetEntries.length + ' files beside ' + output.archive_xml_name
    + (unreferenced ? ', ' + unreferenced + ' of them not referenced' : '')
  );
  archiveSection.hidden = !assetEntries.length;
}

function setUpXmlDownload(content, output, file) {
  if (!content || !content.entries) {
    downloadXmlButton.hidden = true;
    downloadButton.textContent = 'Download';
    return;
  }
  downloadXmlButton.hidden = false;
  downloadXmlButton.textContent = 'Download ' + output.archive_xml_name;
  downloadButton.textContent = 'Download the archive';
  downloadXmlButton.onclick = () => {
    const url = URL.createObjectURL(
      new Blob([content.text], { type: 'application/xml' })
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = (
      file.name.replace(/\.[^.]+$/, '') + '.' + output.archive_xml_name
    );
    link.click();
    URL.revokeObjectURL(url);
  };
}

// The name asked for leads, since that is the one the reader chose; the profile
// that answered follows it, because that is the one that stays meaningful.
function describeServingProfile(response) {
  const served = response.headers.get(PROFILE_HEADER);
  const requested = response.headers.get(PROFILE_REQUESTED_HEADER);
  if (!served) {
    return requested || null;
  }
  return requested ? requested + ' (served by ' + served + ')' : served;
}

function showResultSummary(response, blob, elapsedSeconds) {
  resultSummary.textContent = [
    describeServingProfile(response) || 'the default profile',
    describeBytes(blob.size),
    elapsedSeconds + 's'
  ].join(' · ');
}

async function showResult(response, blob, file, elapsedSeconds) {
  showResultSummary(response, blob, elapsedSeconds);
  const metaElement = document.getElementById('result-meta');
  metaElement.replaceChildren();
  addMetaRow(metaElement, 'Profile', describeServingProfile(response));
  addMetaRow(metaElement, 'Profile digest', response.headers.get(PROFILE_DIGEST_HEADER));
  addMetaRow(metaElement, 'LLM usage', response.headers.get(LLM_USAGE_HEADER));
  addMetaRow(metaElement, 'Content type', response.headers.get('content-type'));
  addMetaRow(metaElement, 'Size', describeBytes(blob.size));
  addMetaRow(metaElement, 'Took', elapsedSeconds + 's');
  const output = getSelectedOutput();
  resultNote.textContent = '';
  let content = null;
  if (output && output.is_text) {
    content = { text: await blob.text(), entries: null };
  } else if (output && output.archive_xml_name) {
    content = await getArchiveContentOrNote(blob, output);
  }
  resultBody.textContent = content
    ? content.text
    : 'Not text. Download it to read it.';
  if (content && content.entries) {
    resultNote.textContent = (
      'Showing ' + output.archive_xml_name + ' from the archive, which carries '
      + content.entries.length + ' files.'
    );
  }
  let viewModel = null;
  if (content && output.reading) {
    viewModel = await showReadableResult(content, output);
  } else {
    showUnreadableResult(output, !!content);
  }
  showArchiveFiles(content, viewModel, output);
  setUpXmlDownload(content, output, file);
  document.getElementById('result-curl').textContent = getEquivalentCurlCommand(file);
  setUpDownload(blob, getDownloadFilename(file));
  resultSection.hidden = false;
}

async function showFailure(response, file) {
  const metaElement = document.getElementById('failure-meta');
  metaElement.replaceChildren();
  addMetaRow(metaElement, 'Status', response.status + ' ' + response.statusText);
  addMetaRow(metaElement, 'Profile', describeServingProfile(response));
  const body = await response.text();
  let detail = body;
  try {
    detail = JSON.parse(body).detail || body;
  } catch (error) {
    detail = body;
  }
  document.getElementById('failure-body').textContent = (
    detail || 'The response had no body.'
  );
  document.getElementById('failure-curl').textContent = getEquivalentCurlCommand(file);
  failureSection.hidden = false;
}

function showRequestError(message, file) {
  const metaElement = document.getElementById('failure-meta');
  metaElement.replaceChildren();
  addMetaRow(metaElement, 'Status', 'no response');
  document.getElementById('failure-body').textContent = message;
  document.getElementById('failure-curl').textContent = getEquivalentCurlCommand(file);
  failureSection.hidden = false;
}

function startElapsedStatus() {
  const startedAt = Date.now();
  const update = () => {
    const seconds = Math.round((Date.now() - startedAt) / 1000);
    statusElement.textContent = seconds < SLOW_REQUEST_SECONDS
      ? 'Converting, ' + seconds + 's'
      : (
        'Converting, ' + seconds + 's. The first conversion after a start also '
        + 'loads the models, which can take minutes.'
      );
  };
  update();
  const timer = window.setInterval(update, 1000);
  return {
    stop: () => {
      window.clearInterval(timer);
      return Math.round((Date.now() - startedAt) / 1000);
    }
  };
}

async function convert(file) {
  const formData = new FormData();
  formData.append('input', file);
  requestController = new AbortController();
  const elapsed = startElapsedStatus();
  convertButton.disabled = true;
  stopButton.hidden = false;
  try {
    const response = await fetch(getRequestUrl(), {
      method: 'POST',
      headers: { Accept: outputSelect.value },
      body: formData,
      signal: requestController.signal
    });
    if (!response.ok) {
      elapsed.stop();
      statusElement.textContent = '';
      await showFailure(response, file);
      return;
    }
    const blob = await getResponseBlob(response);
    const elapsedSeconds = elapsed.stop();
    statusElement.textContent = '';
    await showResult(response, blob, file, elapsedSeconds);
    showSource(file);
  } catch (error) {
    elapsed.stop();
    if (error.name === 'AbortError') {
      statusElement.textContent = (
        'Stopped waiting. The service is still converting it.'
      );
      return;
    }
    statusElement.textContent = '';
    showRequestError('The request did not complete: ' + error.message, file);
  } finally {
    requestController = null;
    convertButton.disabled = false;
    stopButton.hidden = true;
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const file = getSelectedFile();
  if (!file || !isAcceptedFile(file)) {
    checkSelectedFile();
    return;
  }
  resultSection.hidden = true;
  failureSection.hidden = true;
  sourceSection.hidden = true;
  releaseDownloadUrl();
  releaseSourceUrl();
  releaseAssetUrls();
  Object.keys(assetUrlByEntryName).forEach(
    (name) => delete assetUrlByEntryName[name]
  );
  convert(file);
});

stopButton.addEventListener('click', () => {
  if (requestController) {
    requestController.abort();
  }
});

sourceFileInput.addEventListener('change', checkSelectedFile);
sourceSection.addEventListener('toggle', () => {
  if (sourceSection.open) {
    loadSourceFrame();
  }
});
readingButton.addEventListener('click', () => showReading(true));
rawButton.addEventListener('click', () => showReading(false));

setUpLabels();
setUpSourceFileInput();
setUpOutputSelect();
setUpIncludes();
loadProfiles();
