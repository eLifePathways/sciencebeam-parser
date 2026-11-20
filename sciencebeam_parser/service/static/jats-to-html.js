/*
 * Reading a JATS article: what it says, what each value was labelled as,
 * which one it is and where in the document it was found.
 *
 * It takes plain objects -- `{name, attributes, children}` -- rather than DOM
 * nodes, so what an article reads like can be tested without a browser, and
 * so a consumer rendering with something other than the DOM gets the same
 * reading. What it produces is drawn by `article-view.js`, which knows no
 * vocabulary of its own.
 */
import {
  addPaths,
  getAttribute,
  getChildElements,
  getChildText,
  getDescendantElements,
  getFirstChildElement,
  getFirstDescendantElement,
  getInlineRuns,
  getTextContent,
  getValue,
  getLanguage,
  getLanguageValue,
  getValuesWithLanguage,
  parseXmlNode
} from './xml-nodes.js';
import { renderArticleViewModel } from './article-view.js';

const AFFILIATION_PARTS = [
  ['institution', 'orgdiv2'],
  ['institution', 'orgdiv1'],
  ['institution', 'orgname'],
  ['addr-line', null],
  ['city', null],
  ['state', null],
  ['postal-code', null],
  ['country', null]
];

const REFERENCE_FIELDS = [
  ['articleTitle', 'article-title'],
  ['source', 'source'],
  ['year', 'year'],
  ['volume', 'volume'],
  ['issue', 'issue'],
  ['firstPage', 'fpage'],
  ['lastPage', 'lpage']
];

const JATS_RUN_STYLES = {
  bold: { bold: true },
  italic: { italic: true },
  sub: { subscript: true },
  sup: { superscript: true }
};

function getJatsInline(node) {
  if (JATS_RUN_STYLES[node.name]) {
    return { style: JATS_RUN_STYLES[node.name] };
  }
  if (node.name === 'xref') {
    return { leaf: { reference: getAttribute(node, 'rid') } };
  }
  return null;
}

function isStyled(run) {
  return run.bold || run.italic || run.subscript || run.superscript;
}

function trimRuns(runs) {
  const trimmed = runs.map((run) => ({ ...run }));
  trimmed[0].text = trimmed[0].text.trimStart();
  trimmed[trimmed.length - 1].text = trimmed[trimmed.length - 1].text.trimEnd();
  return trimmed.filter((run) => run.text !== '');
}

/* The value, with its runs where any of them is formatted. */
function withRuns(value, element) {
  if (!value || !element) {
    return value;
  }
  const runs = getInlineRuns(element, getJatsInline);
  return runs.some(isStyled) ? { ...value, runs: trimRuns(runs) } : value;
}

function getFormattedValue(node, name) {
  return withRuns(getValue(node, name), getFirstChildElement(node, name));
}

function getParagraph(node) {
  return {
    label: node.name, path: node.path, runs: getInlineRuns(node, getJatsInline)
  };
}

function getCaption(node) {
  const caption = getFirstChildElement(node, 'caption');
  return {
    label: node.name,
    path: node.path,
    heading: getValue(node, 'label'),
    caption: caption ? { label: caption.name, path: caption.path } : null,
    title: caption ? getFormattedValue(caption, 'title') : null,
    paragraphs: caption ? getChildElements(caption, 'p').map(getParagraph) : []
  };
}

function getGraphics(node) {
  return getChildElements(node, 'graphic').map((graphic) => ({
    label: graphic.name,
    path: graphic.path,
    href: getAttribute(graphic, 'xlink:href')
  }));
}

function getFigure(node) {
  return {
    type: 'figure',
    id: getAttribute(node, 'id'),
    ...getCaption(node),
    graphics: getGraphics(node)
  };
}

function getTable(node) {
  const table = getFirstChildElement(node, 'table');
  const body = table ? getFirstChildElement(table, 'tbody') : null;
  return {
    type: 'table',
    id: getAttribute(node, 'id'),
    ...getCaption(node),
    graphics: getGraphics(node),
    rows: (body ? getChildElements(body, 'tr') : []).map(
      (row) => getChildElements(row, 'td').map(getParagraph)
    )
  };
}

function getSectionBlocks(node) {
  const blocks = [];
  getChildElements(node).forEach((child) => {
    if (child.name === 'p') {
      blocks.push({ type: 'paragraph', ...getParagraph(child) });
    } else if (child.name === 'fig') {
      blocks.push(getFigure(child));
    } else if (child.name === 'table-wrap') {
      blocks.push(getTable(child));
    } else if (child.name === 'sec') {
      blocks.push(getSection(child));
    }
  });
  return blocks;
}

function getSection(node) {
  return {
    type: 'section',
    label: node.name,
    path: node.path,
    id: getAttribute(node, 'id'),
    heading: getValue(node, 'label'),
    title: getFormattedValue(node, 'title'),
    blocks: getSectionBlocks(node)
  };
}

function getPersonNameParts(node) {
  const name = getFirstChildElement(node, 'name');
  if (!name) {
    const text = getTextContent(node).trim();
    return text ? [{ label: node.name, text, path: node.path }] : [];
  }
  return [
    getValue(name, 'given-names'),
    getValue(name, 'surname')
  ].filter((part) => part);
}

function getContribGroup(articleMeta) {
  const contribGroup = getFirstChildElement(articleMeta, 'contrib-group');
  return contribGroup
    ? { label: contribGroup.name, path: contribGroup.path }
    : null;
}

function getAuthors(articleMeta) {
  const contribGroup = getFirstChildElement(articleMeta, 'contrib-group');
  return getChildElements(contribGroup, 'contrib').map((contrib) => ({
    label: contrib.name,
    path: contrib.path,
    nameParts: getPersonNameParts(contrib),
    email: getValue(contrib, 'email'),
    affiliationRefs: getChildElements(contrib, 'xref')
      .filter((xref) => getAttribute(xref, 'ref-type') === 'aff')
      .map((xref) => ({
        label: xref.name, path: xref.path, rid: getAttribute(xref, 'rid')
      }))
      .filter((ref) => ref.rid)
  }));
}

function getAffiliationPart(child) {
  return {
    label: getAttribute(child, 'content-type') || child.name,
    path: child.path,
    text: getTextContent(child).trim()
  };
}

function getAffiliationParts(aff) {
  return AFFILIATION_PARTS.flatMap(([name, contentType]) => getChildElements(aff, name)
    .filter(
      (child) => !contentType
        || getAttribute(child, 'content-type') === contentType
    )
    .map(getAffiliationPart)
    .filter((part) => part.text));
}

function isPrintedText(aff) {
  return (aff.children || []).some(
    (child) => typeof child === 'string' && /[^\s,]/.test(child)
  );
}

/* The affiliation as the page printed it, where the document carries that. */
function getAffiliationContent(aff) {
  if (!isPrintedText(aff)) {
    return null;
  }
  const content = [];
  (aff.children || []).forEach((child) => {
    if (typeof child === 'string') {
      content.push({ text: child });
    } else if (child.name !== 'label') {
      content.push(getAffiliationPart(child));
    }
  });
  return content;
}

function getAffiliations(articleMeta) {
  const contribGroup = getFirstChildElement(articleMeta, 'contrib-group');
  return getChildElements(contribGroup, 'aff').map((aff) => ({
    label: aff.name,
    path: aff.path,
    id: getAttribute(aff, 'id'),
    heading: getValue(aff, 'label'),
    parts: getAffiliationParts(aff),
    content: getAffiliationContent(aff)
  }));
}

/* What a citation tags, where it stands in the citation's text: a name's
 * parts each stand for themselves. */
function getCitationContent(node) {
  return (node.children || []).flatMap((child) => {
    if (typeof child === 'string') {
      return [{ text: child }];
    }
    if (child.name === 'person-group' || child.name === 'string-name') {
      return getCitationContent(child);
    }
    const value = withRuns(
      { label: child.name, path: child.path, text: getTextContent(child) }, child
    );
    const href = getAttribute(child, 'xlink:href');
    return [href ? { ...value, href } : value];
  });
}

function getReference(node) {
  const mixedCitation = getFirstChildElement(node, 'mixed-citation');
  const citation = getFirstChildElement(node, 'element-citation') || mixedCitation;
  const personGroup = citation
    ? getFirstChildElement(citation, 'person-group')
    : null;
  const pubIds = citation ? getChildElements(citation, 'pub-id') : [];
  const doi = pubIds.find((pubId) => getAttribute(pubId, 'pub-id-type') === 'doi');
  const url = citation ? getFirstChildElement(citation, 'ext-link') : null;
  const reference = {
    label: node.name,
    path: node.path,
    id: getAttribute(node, 'id'),
    heading: getValue(node, 'label'),
    authors: [
      ...getChildElements(personGroup, 'name'),
      ...getChildElements(personGroup, 'string-name')
    ].map((name) => [
      getValue(name, 'given-names'),
      getValue(name, 'surname')
    ].filter((part) => part)),
    content: mixedCitation ? getCitationContent(mixedCitation) : null,
    doi: doi
      ? { label: 'pub-id', path: doi.path, text: getTextContent(doi).trim() }
      : null,
    url: url
      ? {
        label: 'ext-link',
        path: url.path,
        text: getTextContent(url).trim(),
        href: getAttribute(url, 'xlink:href')
      }
      : null
  };
  REFERENCE_FIELDS.forEach(([key, name]) => {
    reference[key] = citation ? getFormattedValue(citation, name) : null;
  });
  return reference;
}

/* A value in the language it declares, or else the one its container declares. */
function withLanguage(value, ...containers) {
  const language = [value.language, ...containers.map(getLanguage)].find((each) => each);
  return language ? { ...value, language } : value;
}

function getAbstract(abstract, article) {
  const language = getLanguage(abstract) || getLanguage(article);
  const value = {
    label: abstract.name,
    path: abstract.path,
    title: getFormattedValue(abstract, 'title'),
    paragraphs: getDescendantElements(abstract, 'p').map(getParagraph)
  };
  return language ? { ...value, language } : value;
}


export function getArticleViewModel(article) {
  if (!article.path) {
    article.path = `/${article.name}`;
    addPaths(article, article.path);
  }
  const front = getFirstChildElement(article, 'front');
  const articleMeta = getFirstChildElement(front, 'article-meta');
  const body = getFirstChildElement(article, 'body');
  const back = getFirstChildElement(article, 'back');
  const ack = getFirstChildElement(back, 'ack');
  const appGroup = getFirstChildElement(back, 'app-group');
  const refList = back ? getFirstDescendantElement(back, 'ref-list') : null;
  const journalTitle = front
    ? getFirstDescendantElement(front, 'journal-title')
    : null;
  return {
    titles: getChildElements(articleMeta, 'title-group').flatMap(
      (group) => getValuesWithLanguage(group, 'article-title').map(
        (value) => withLanguage(withRuns(
          value,
          getChildElements(group, 'article-title').find((title) => title.path === value.path)
        ), group, article)
      )
    ),
    language: getLanguageValue(article),
    journal: journalTitle
      ? {
        label: 'journal-title',
        path: journalTitle.path,
        text: getTextContent(journalTitle).trim()
      }
      : null,
    authorGroup: getContribGroup(articleMeta),
    authors: getAuthors(articleMeta),
    affiliations: getAffiliations(articleMeta),
    abstracts: getChildElements(articleMeta, 'abstract').map(
      (abstract) => getAbstract(abstract, article)
    ),
    sections: getChildElements(body, 'sec').map(getSection),
    acknowledgements: ack ? getSectionBlocks(ack) : [],
    appendices: getChildElements(appGroup, 'app').map(getSection),
    references: getChildElements(refList, 'ref').map(getReference)
  };
}

export function parseJatsViewModel(xmlText) {
  return getArticleViewModel(parseXmlNode(xmlText));
}

