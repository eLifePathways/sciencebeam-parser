const deployment = JSON.parse(document.getElementById('deployment').textContent);

const PROFILE_HEADER = 'x-sciencebeam-profile';
const PROFILE_DIGEST_HEADER = 'x-sciencebeam-profile-digest';
const LLM_USAGE_HEADER = 'x-sciencebeam-llm-usage';
const SLOW_REQUEST_SECONDS = 10;

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

let downloadUrl = null;
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

function showSelectedOutputHint() {
  const output = getSelectedOutput();
  outputHint.textContent = output && !output.is_text
    ? 'Returned as a file to download rather than as text to read here.'
    : '';
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
  document.getElementById('result-body').textContent = output && output.is_text
    ? await blob.text()
    : 'Not text. Download it to read it.';
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
  releaseDownloadUrl();
  convert(file);
});

stopButton.addEventListener('click', () => {
  if (requestController) {
    requestController.abort();
  }
});

sourceFileInput.addEventListener('change', checkSelectedFile);

setUpSourceFileInput();
setUpOutputSelect();
setUpIncludes();
loadProfiles();
