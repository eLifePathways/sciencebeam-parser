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
  parseXmlNode
} from './xml-nodes.js';
import { renderArticleViewModel } from './article-view.js';

const AFFILIATION_PARTS = [
  ['institution', 'orgdiv2'],
  ['institution', 'orgdiv1'],
  ['institution', 'orgname'],
  ['addr-line', null],
  ['city', null],
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

const JATS_RUN_STYLES = { b: { bold: true }, i: { italic: true } };

function getJatsInline(node) {
  if (JATS_RUN_STYLES[node.name]) {
    return { style: JATS_RUN_STYLES[node.name] };
  }
  if (node.name === 'xref') {
    return { leaf: { reference: getAttribute(node, 'rid') } };
  }
  return null;
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
    title: caption ? getValue(caption, 'title') : null,
    paragraphs: caption ? getChildElements(caption, 'p').map(getParagraph) : []
  };
}

function getFigure(node) {
  return {
    type: 'figure',
    id: getAttribute(node, 'id'),
    ...getCaption(node),
    graphics: getChildElements(node, 'graphic').map((graphic) => ({
      label: graphic.name,
      path: graphic.path,
      href: getAttribute(graphic, 'xlink:href')
    }))
  };
}

function getTable(node) {
  const table = getFirstChildElement(node, 'table');
  const body = table ? getFirstChildElement(table, 'tbody') : null;
  return {
    type: 'table',
    id: getAttribute(node, 'id'),
    ...getCaption(node),
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
    title: getValue(node, 'title'),
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

function getAffiliations(articleMeta) {
  return getChildElements(articleMeta, 'aff').map((aff) => ({
    label: aff.name,
    path: aff.path,
    id: getAttribute(aff, 'id'),
    parts: AFFILIATION_PARTS.flatMap(([name, contentType]) => getChildElements(aff, name)
      .filter(
        (child) => !contentType
          || getAttribute(child, 'content-type') === contentType
      )
      .map((child) => ({
        label: contentType || name,
        path: child.path,
        text: getTextContent(child).trim()
      }))
      .filter((part) => part.text))
  }));
}

function getReference(node) {
  const citation = getFirstChildElement(node, 'element-citation')
    || getFirstChildElement(node, 'mixed-citation');
  const personGroup = citation
    ? getFirstChildElement(citation, 'person-group')
    : null;
  const pubIds = citation ? getChildElements(citation, 'pub-id') : [];
  const doi = pubIds.find((pubId) => getAttribute(pubId, 'pub-id-type') === 'doi');
  const reference = {
    label: node.name,
    path: node.path,
    id: getAttribute(node, 'id'),
    authors: getChildElements(personGroup, 'name').map((name) => [
      getValue(name, 'given-names'),
      getValue(name, 'surname')
    ].filter((part) => part)),
    doi: doi
      ? { label: 'pub-id', path: doi.path, text: getTextContent(doi).trim() }
      : null
  };
  REFERENCE_FIELDS.forEach(([key, name]) => {
    reference[key] = citation ? getValue(citation, name) : null;
  });
  return reference;
}

export function getArticleViewModel(article) {
  if (!article.path) {
    article.path = `/${article.name}`;
    addPaths(article, article.path);
  }
  const front = getFirstChildElement(article, 'front');
  const articleMeta = getFirstChildElement(front, 'article-meta');
  const titleGroup = getFirstChildElement(articleMeta, 'title-group');
  const body = getFirstChildElement(article, 'body');
  const back = getFirstChildElement(article, 'back');
  const ack = getFirstChildElement(back, 'ack');
  const appGroup = getFirstChildElement(back, 'app-group');
  const refList = back ? getFirstDescendantElement(back, 'ref-list') : null;
  const journalTitle = front
    ? getFirstDescendantElement(front, 'journal-title')
    : null;
  return {
    title: titleGroup ? getValue(titleGroup, 'article-title') : null,
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
    abstracts: getChildElements(articleMeta, 'abstract').map((abstract) => ({
      label: abstract.name,
      path: abstract.path,
      title: getValue(abstract, 'title'),
      paragraphs: getDescendantElements(abstract, 'p').map(getParagraph)
    })),
    sections: getChildElements(body, 'sec').map(getSection),
    acknowledgements: ack ? getSectionBlocks(ack) : [],
    appendices: getChildElements(appGroup, 'app').map(getSection),
    references: getChildElements(refList, 'ref').map(getReference)
  };
}

export function parseJatsViewModel(xmlText) {
  return getArticleViewModel(parseXmlNode(xmlText));
}

