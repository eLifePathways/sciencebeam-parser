import { renderJatsXml } from '/static/jats-to-html.js';
import { readZipEntries } from '/static/read-zip.js';

const deployment = JSON.parse(document.getElementById('deployment').textContent);

const PROFILE_HEADER = 'x-sciencebeam-profile';
const PROFILE_DIGEST_HEADER = 'x-sciencebeam-profile-digest';
const LLM_USAGE_HEADER = 'x-sciencebeam-llm-usage';
const SLOW_REQUEST_SECONDS = 10;
const PDF_MEDIA_TYPE = 'application/pdf';
const LABELS_SETTING = 'sciencebeam-show-labels';

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
  if (!output || output.is_readable) {
    return '';
  }
  if (output.is_text || output.archive_xml_name) {
    return 'Returned as the XML itself. The reading view is over JATS XML.';
  }
  return 'Returned as a file to download rather than as text to read here.';
}

function showSelectedOutputHint() {
  outputHint.textContent = getOutputHint(getSelectedOutput());
}

function setUpIncludes() {
  deployment.includes_field_names.forEach((fieldName) => {
    const label = document.createElement('label');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = fieldName;
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(' ' + fieldName));
    includesElement.appendChild(label);
  });
}

function getSelectedIncludes() {
  return Array.from(includesElement.querySelectorAll('input:checked'))
    .map((checkbox) => checkbox.value);
}

function showProfileDescription(profiles) {
  const profile = profiles.find((item) => item.name === profileSelect.value);
  profileHint.textContent = profile ? (profile.description || '') : '';
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
  profiles.forEach((profile) => {
    const option = addOption(profileSelect, profile.name, profile.label || profile.name);
    option.selected = profile.is_default;
  });
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

function getAssetUrlByName(entries) {
  const urlByName = {};
  entries.forEach((entry) => {
    const mediaType = getImageMediaType(entry.name);
    if (!mediaType) {
      return;
    }
    const url = URL.createObjectURL(new Blob([entry.data], { type: mediaType }));
    assetUrls.push(url);
    urlByName[entry.name] = url;
  });
  return urlByName;
}

function releaseSourceUrl() {
  if (sourceUrl) {
    URL.revokeObjectURL(sourceUrl);
    sourceUrl = null;
  }
}

function showSourceBlob(blob) {
  releaseSourceUrl();
  sourceUrl = URL.createObjectURL(blob);
  sourceFrame.src = sourceUrl;
  sourceFrame.hidden = false;
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
  readingButton.disabled = isReading;
  rawButton.disabled = !isReading;
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

function showReadableResult(xmlText, getAssetUrl) {
  readingElement.replaceChildren();
  try {
    readingElement.appendChild(
      renderJatsXml(xmlText, document, { getAssetUrl })
    );
  } catch (error) {
    resultNote.textContent = 'It could not be read as JATS: ' + error.message;
    readingButton.hidden = true;
    rawButton.hidden = true;
    showReading(false);
    return;
  }
  readingButton.hidden = false;
  rawButton.hidden = false;
  showReading(true);
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
  const entries = await readZipEntries(await blob.arrayBuffer());
  const xmlEntry = entries.find((entry) => entry.name === output.archive_xml_name);
  if (!xmlEntry) {
    throw new Error(
      'the archive carries no ' + output.archive_xml_name
    );
  }
  const assetUrlByName = getAssetUrlByName(entries);
  return {
    text: new TextDecoder().decode(xmlEntry.data),
    entryCount: entries.length,
    getAssetUrl: (href) => assetUrlByName[href] || null
  };
}

async function showResult(response, blob, file, elapsedSeconds) {
  const metaElement = document.getElementById('result-meta');
  metaElement.replaceChildren();
  addMetaRow(metaElement, 'Served by profile', response.headers.get(PROFILE_HEADER));
  addMetaRow(metaElement, 'Profile digest', response.headers.get(PROFILE_DIGEST_HEADER));
  addMetaRow(metaElement, 'LLM usage', response.headers.get(LLM_USAGE_HEADER));
  addMetaRow(metaElement, 'Content type', response.headers.get('content-type'));
  addMetaRow(metaElement, 'Size', blob.size.toLocaleString() + ' bytes');
  addMetaRow(metaElement, 'Took', elapsedSeconds + 's');
  const output = getSelectedOutput();
  resultNote.textContent = '';
  let content = null;
  if (output && output.is_text) {
    content = { text: await blob.text() };
  } else if (output && output.archive_xml_name) {
    content = await getArchiveContentOrNote(blob, output);
  }
  resultBody.textContent = content
    ? content.text
    : 'Not text. Download it to read it.';
  if (content && output.archive_xml_name) {
    resultNote.textContent = (
      'Showing ' + output.archive_xml_name + ' from the archive, which carries '
      + content.entryCount + ' files. The download is the archive itself.'
    );
  }
  if (content && output.is_readable) {
    showReadableResult(content.text, content.getAssetUrl);
  } else {
    showUnreadableResult(output, !!content);
  }
  document.getElementById('result-curl').textContent = getEquivalentCurlCommand(file);
  setUpDownload(blob, getDownloadFilename(file));
  resultSection.hidden = false;
}

async function showFailure(response, file) {
  const metaElement = document.getElementById('failure-meta');
  metaElement.replaceChildren();
  addMetaRow(metaElement, 'Status', response.status + ' ' + response.statusText);
  addMetaRow(metaElement, 'Served by profile', response.headers.get(PROFILE_HEADER));
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
    const elapsedSeconds = elapsed.stop();
    statusElement.textContent = '';
    if (!response.ok) {
      await showFailure(response, file);
      return;
    }
    await showResult(response, await response.blob(), file, elapsedSeconds);
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
  convert(file);
});

stopButton.addEventListener('click', () => {
  if (requestController) {
    requestController.abort();
  }
});

sourceFileInput.addEventListener('change', checkSelectedFile);
readingButton.addEventListener('click', () => showReading(true));
rawButton.addEventListener('click', () => showReading(false));

setUpLabels();
setUpSourceFileInput();
setUpOutputSelect();
setUpIncludes();
loadProfiles();
