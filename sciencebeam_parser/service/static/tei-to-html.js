/*
 * Reading a TEI article: what it says, what each value was labelled as, which
 * one it is and where in the document it was found.
 *
 * The same reading as JATS gets, from a document that says it differently. An
 * author carries its own affiliations rather than pointing at a list of them,
 * a section is a div with a head, a table is a figure that says it is one,
 * and emphasis is a rend rather than an element of its own. What it produces
 * is drawn by `article-view.js`, which knows neither vocabulary.
 */
import {
  addPaths,
  getAttribute,
  getChildElements,
  getFirstChildElement,
  getFirstDescendantElement,
  getInlineRuns,
  getTextContent,
  getValue,
  parseXmlNode
} from './xml-nodes.js';
import { renderArticleViewModel } from './article-view.js';

const TEI_RUN_STYLES = {
  bold: { bold: true },
  italic: { italic: true },
  subscript: { subscript: true },
  superscript: { superscript: true }
};

// Smallest outwards, as an address is written.
const AFFILIATION_PARTS = [
  ['orgName', 'laboratory'],
  ['orgName', 'department'],
  ['orgName', 'institution'],
  ['addrLine', null],
  ['settlement', null],
  ['postCode', null],
  ['country', null]
];

const REFERENCE_SCOPES = [
  ['volume', 'volume'],
  ['issue', 'issue']
];

function getTeiInline(node) {
  if (node.name === 'hi') {
    return { style: TEI_RUN_STYLES[getAttribute(node, 'rend')] || {} };
  }
  if (node.name === 'ref') {
    return { leaf: { reference: getAttribute(node, 'target').replace(/^#/, '') } };
  }
  return null;
}

function getElementId(node) {
  return getAttribute(node, 'xml:id') || getAttribute(node, 'id');
}

function getParagraph(node) {
  return {
    label: node.name, path: node.path, runs: getInlineRuns(node, getTeiInline)
  };
}

function getChildByAttribute(node, name, attributeName, attributeValue) {
  return getChildElements(node, name).find(
    (child) => getAttribute(child, attributeName) === attributeValue
  ) || null;
}

function getNodeValue(node, label) {
  if (!node) {
    return null;
  }
  const text = getTextContent(node).trim();
  return text ? { label: label || node.name, text, path: node.path } : null;
}

function getFigure(node) {
  const isTable = getAttribute(node, 'type') === 'table';
  return {
    type: isTable ? 'table' : 'figure',
    label: node.name,
    path: node.path,
    id: getElementId(node),
    heading: getValue(node, 'label'),
    title: getValue(node, 'head'),
    paragraphs: getChildElements(node, 'figDesc').map(getParagraph),
    rows: [],
    graphics: getChildElements(node, 'graphic').map((graphic) => ({
      label: graphic.name,
      path: graphic.path,
      href: getAttribute(graphic, 'url')
    }))
  };
}

function getDivisionBlocks(node) {
  const blocks = [];
  getChildElements(node).forEach((child) => {
    if (child.name === 'p') {
      blocks.push({ type: 'paragraph', ...getParagraph(child) });
    } else if (child.name === 'figure') {
      blocks.push(getFigure(child));
    } else if (child.name === 'div') {
      blocks.push(getDivision(child));
    }
  });
  return blocks;
}

function getDivision(node) {
  return {
    type: 'section',
    label: node.name,
    path: node.path,
    id: getElementId(node),
    title: getValue(node, 'head'),
    blocks: getDivisionBlocks(node)
  };
}

function getPersonNameParts(node) {
  const name = getFirstChildElement(node, 'persName') || node;
  const parts = getChildElements(name, 'forename').map(
    (forename) => getNodeValue(forename)
  );
  parts.push(getNodeValue(getFirstChildElement(name, 'surname')));
  return parts.filter((part) => part);
}

function getAffiliationParts(affiliation) {
  const address = getFirstChildElement(affiliation, 'address');
  return AFFILIATION_PARTS.flatMap(([name, type]) => (
    [affiliation, address]
      .filter((parent) => parent)
      .flatMap((parent) => getChildElements(parent, name))
      .filter((child) => !type || getAttribute(child, 'type') === type)
      .map((child) => getNodeValue(child, type || name))
      .filter((part) => part)
  ));
}

function getAffiliation(node) {
  return {
    label: node.name,
    path: node.path,
    id: getAttribute(node, 'key'),
    parts: getAffiliationParts(node)
  };
}

function getAuthorElements(analytic) {
  return getChildElements(analytic, 'author');
}

function getAuthor(node) {
  return {
    label: node.name,
    path: node.path,
    nameParts: getPersonNameParts(node),
    email: getValue(node, 'email')
      || getValue(getFirstChildElement(node, 'persName') || node, 'email'),
    // A TEI author carries its affiliations rather than pointing at them, so
    // what stands for the link is the affiliation itself.
    affiliationRefs: getChildElements(node, 'affiliation').map((affiliation) => ({
      label: affiliation.name,
      path: affiliation.path,
      rid: getAttribute(affiliation, 'key')
    }))
  };
}

function getReferencePages(imprint) {
  const page = getChildByAttribute(imprint, 'biblScope', 'unit', 'page');
  if (!page) {
    return { firstPage: null, lastPage: null };
  }
  const from = getAttribute(page, 'from');
  const to = getAttribute(page, 'to');
  if (!from && !to) {
    return { firstPage: getNodeValue(page, 'page'), lastPage: null };
  }
  return {
    firstPage: from ? { label: 'page', text: from, path: page.path } : null,
    lastPage: to ? { label: 'page', text: to, path: page.path } : null
  };
}

function getReference(node) {
  const analytic = getFirstChildElement(node, 'analytic');
  const monogr = getFirstChildElement(node, 'monogr');
  const imprint = monogr ? getFirstChildElement(monogr, 'imprint') : null;
  const authorParent = analytic && getAuthorElements(analytic).length
    ? analytic
    : monogr;
  const doi = node
    ? getChildByAttribute(node, 'idno', 'type', 'DOI')
    : null;
  const reference = {
    label: node.name,
    path: node.path,
    id: getElementId(node),
    authors: getAuthorElements(authorParent || node).map(getPersonNameParts),
    articleTitle: analytic
      ? getNodeValue(
        getChildByAttribute(analytic, 'title', 'level', 'a'), 'title'
      )
      : null,
    source: monogr
      ? getNodeValue(getChildByAttribute(monogr, 'title', 'level', 'j'), 'title')
      : null,
    year: imprint
      ? getNodeValue(getFirstChildElement(imprint, 'date'), 'date')
      : null,
    doi: getNodeValue(doi, 'idno'),
    ...getReferencePages(imprint || node)
  };
  REFERENCE_SCOPES.forEach(([key, unit]) => {
    reference[key] = imprint
      ? getNodeValue(
        getChildByAttribute(imprint, 'biblScope', 'unit', unit), 'biblScope'
      )
      : null;
  });
  return reference;
}

function getBackDivision(back, type) {
  return getChildElements(back, 'div').find(
    (division) => getAttribute(division, 'type') === type
  ) || null;
}

export function getTeiArticleViewModel(article) {
  if (!article.path) {
    article.path = `/${article.name}`;
    addPaths(article, article.path);
  }
  const header = getFirstChildElement(article, 'teiHeader');
  const titleStmt = header
    ? getFirstDescendantElement(header, 'titleStmt')
    : null;
  const analytic = header ? getFirstDescendantElement(header, 'analytic') : null;
  const text = getFirstChildElement(article, 'text');
  const body = text ? getFirstChildElement(text, 'body') : null;
  const back = text ? getFirstChildElement(text, 'back') : null;
  const abstract = header
    ? getFirstDescendantElement(header, 'abstract')
    : null;
  const journalTitle = header
    ? getChildByAttribute(
      getFirstDescendantElement(header, 'monogr') || {}, 'title', 'level', 'j'
    )
    : null;
  const authors = analytic ? getAuthorElements(analytic) : [];
  const annex = back ? getBackDivision(back, 'annex') : null;
  const acknowledgement = back
    ? getBackDivision(back, 'acknowledgement')
    : null;
  const references = back ? getBackDivision(back, 'references') : null;
  const listBibl = references
    ? getFirstDescendantElement(references, 'listBibl')
    : null;
  return {
    title: titleStmt ? getValue(titleStmt, 'title') : null,
    journal: getNodeValue(journalTitle, 'title'),
    authorGroup: analytic
      ? { label: analytic.name, path: analytic.path }
      : null,
    authors: authors.map(getAuthor),
    affiliations: authors.flatMap(
      (author) => getChildElements(author, 'affiliation').map(getAffiliation)
    ),
    abstracts: abstract
      ? [{
        label: abstract.name,
        path: abstract.path,
        title: null,
        paragraphs: getChildElements(abstract, 'p').map(getParagraph)
      }]
      : [],
    sections: body ? getDivisionBlocks(body) : [],
    acknowledgements: acknowledgement
      ? getDivisionBlocks(acknowledgement)
      : [],
    appendices: annex ? getDivisionBlocks(annex) : [],
    references: listBibl
      ? getChildElements(listBibl, 'biblStruct').map(getReference)
      : []
  };
}

export function parseTeiViewModel(xmlText) {
  return getTeiArticleViewModel(parseXmlNode(xmlText));
}

export function renderTeiXml(xmlText, document, options) {
  return renderArticleViewModel(parseTeiViewModel(xmlText), document, options);
}
