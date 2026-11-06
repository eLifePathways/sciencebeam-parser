/*
 * Turning JATS into something a person can read.
 *
 * Nothing here imports anything or assumes the page around it, so that a site
 * other than this one can use the same reading of an article.
 *
 * `getArticleViewModel` is the part that knows JATS. It takes plain objects --
 * `{name, attributes, children}`, children being nodes or strings -- rather
 * than DOM nodes, so what an article reads like can be tested without a
 * browser, and so a consumer rendering with something other than the DOM gets
 * the same reading.
 */

const AFFILIATION_PART_SELECTORS = [
  'institution[content-type="orgdiv2"]',
  'institution[content-type="orgdiv1"]',
  'institution[content-type="orgname"]',
  'addr-line',
  'city',
  'country'
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

function isElement(node) {
  return !!node && typeof node === 'object' && typeof node.name === 'string';
}

function getChildElements(node, name) {
  if (!isElement(node)) {
    return [];
  }
  return (node.children || []).filter(
    (child) => isElement(child) && (!name || child.name === name)
  );
}

function getFirstChildElement(node, name) {
  return getChildElements(node, name)[0] || null;
}

function getDescendantElements(node, name) {
  const found = [];
  getChildElements(node).forEach((child) => {
    if (child.name === name) {
      found.push(child);
    }
    found.push(...getDescendantElements(child, name));
  });
  return found;
}

function getFirstDescendantElement(node, name) {
  return getDescendantElements(node, name)[0] || null;
}

function getAttribute(node, name) {
  return (isElement(node) && (node.attributes || {})[name]) || '';
}

function getTextContent(node) {
  if (typeof node === 'string') {
    return node;
  }
  if (!isElement(node)) {
    return '';
  }
  return (node.children || []).map(getTextContent).join('');
}

function getChildText(node, name) {
  const child = getFirstChildElement(node, name);
  return child ? getTextContent(child).trim() : '';
}

function getInlineRuns(node, style) {
  const currentStyle = style || {};
  const runs = [];
  (isElement(node) ? node.children || [] : []).forEach((child) => {
    if (typeof child === 'string') {
      runs.push({ ...currentStyle, text: child });
      return;
    }
    if (child.name === 'b') {
      runs.push(...getInlineRuns(child, { ...currentStyle, bold: true }));
      return;
    }
    if (child.name === 'i') {
      runs.push(...getInlineRuns(child, { ...currentStyle, italic: true }));
      return;
    }
    if (child.name === 'xref') {
      runs.push({
        ...currentStyle,
        text: getTextContent(child),
        reference: getAttribute(child, 'rid')
      });
      return;
    }
    runs.push(...getInlineRuns(child, currentStyle));
  });
  return runs.filter((run) => run.text !== '');
}

function getParagraph(node) {
  return { runs: getInlineRuns(node) };
}

function getCaption(node) {
  const caption = getFirstChildElement(node, 'caption');
  return {
    label: getChildText(node, 'label'),
    title: caption ? getChildText(caption, 'title') : '',
    paragraphs: caption ? getChildElements(caption, 'p').map(getParagraph) : []
  };
}

function getFigure(node) {
  const graphic = getFirstChildElement(node, 'graphic');
  return {
    type: 'figure',
    id: getAttribute(node, 'id'),
    ...getCaption(node),
    graphicHref: graphic ? getAttribute(graphic, 'xlink:href') : ''
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
    id: getAttribute(node, 'id'),
    title: getChildText(node, 'title'),
    blocks: getSectionBlocks(node)
  };
}

function getPersonName(node) {
  const name = getFirstChildElement(node, 'name');
  if (!name) {
    return getTextContent(node).trim();
  }
  return [getChildText(name, 'given-names'), getChildText(name, 'surname')]
    .filter((part) => part)
    .join(' ');
}

function getAuthors(articleMeta) {
  const contribGroup = getFirstChildElement(articleMeta, 'contrib-group');
  return getChildElements(contribGroup, 'contrib').map((contrib) => ({
    name: getPersonName(contrib),
    email: getChildText(contrib, 'email'),
    affiliationIds: getChildElements(contrib, 'xref')
      .filter((xref) => getAttribute(xref, 'ref-type') === 'aff')
      .map((xref) => getAttribute(xref, 'rid'))
      .filter((rid) => rid)
  }));
}

function getAffiliations(articleMeta) {
  return getChildElements(articleMeta, 'aff').map((aff) => ({
    id: getAttribute(aff, 'id'),
    text: AFFILIATION_PART_SELECTORS
      .map((selector) => {
        const [name, contentType] = selector.split('[content-type="');
        const parts = getChildElements(aff, name).filter(
          (child) => !contentType
            || getAttribute(child, 'content-type') === contentType.slice(0, -2)
        );
        return parts.map((part) => getTextContent(part).trim()).join(', ');
      })
      .filter((part) => part)
      .join(', ')
  }));
}

function getReference(node) {
  const citation = getFirstChildElement(node, 'element-citation')
    || getFirstChildElement(node, 'mixed-citation');
  const personGroup = citation
    ? getFirstChildElement(citation, 'person-group')
    : null;
  const reference = {
    id: getAttribute(node, 'id'),
    authors: getChildElements(personGroup, 'name').map(
      (name) => [getTextContent(getFirstChildElement(name, 'given-names')).trim(),
        getTextContent(getFirstChildElement(name, 'surname')).trim()]
        .filter((part) => part)
        .join(' ')
    ),
    doi: ''
  };
  REFERENCE_FIELDS.forEach(([key, name]) => {
    reference[key] = citation ? getChildText(citation, name) : '';
  });
  const pubIds = citation ? getChildElements(citation, 'pub-id') : [];
  const doi = pubIds.find((pubId) => getAttribute(pubId, 'pub-id-type') === 'doi');
  reference.doi = doi ? getTextContent(doi).trim() : '';
  return reference;
}

export function getArticleViewModel(article) {
  const front = getFirstChildElement(article, 'front');
  const articleMeta = getFirstChildElement(front, 'article-meta');
  const titleGroup = getFirstChildElement(articleMeta, 'title-group');
  const body = getFirstChildElement(article, 'body');
  const back = getFirstChildElement(article, 'back');
  const ack = getFirstChildElement(back, 'ack');
  const appGroup = getFirstChildElement(back, 'app-group');
  const refList = back ? getFirstDescendantElement(back, 'ref-list') : null;
  return {
    title: titleGroup ? getChildText(titleGroup, 'article-title') : '',
    journal: front
      ? getTextContent(getFirstDescendantElement(front, 'journal-title')).trim()
      : '',
    authors: getAuthors(articleMeta),
    affiliations: getAffiliations(articleMeta),
    abstracts: getChildElements(articleMeta, 'abstract').map((abstract) => ({
      title: getChildText(abstract, 'title'),
      paragraphs: getDescendantElements(abstract, 'p').map(getParagraph)
    })),
    sections: getChildElements(body, 'sec').map(getSection),
    acknowledgements: ack ? getSectionBlocks(ack) : [],
    appendices: getChildElements(appGroup, 'app').map(getSection),
    references: getChildElements(refList, 'ref').map(getReference)
  };
}

export function getXmlNode(element) {
  return {
    name: element.localName,
    attributes: Array.from(element.attributes || []).reduce(
      (attributes, attribute) => (
        { ...attributes, [attribute.name]: attribute.value }
      ),
      {}
    ),
    children: Array.from(element.childNodes)
      .map((child) => {
        if (child.nodeType === 3) {
          return child.nodeValue;
        }
        return child.nodeType === 1 ? getXmlNode(child) : null;
      })
      .filter((child) => child !== null)
  };
}

function appendElement(parent, tagName, className, text) {
  const element = parent.ownerDocument.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text) {
    element.textContent = text;
  }
  parent.appendChild(element);
  return element;
}

function appendRuns(parent, runs) {
  runs.forEach((run) => {
    let target = parent;
    if (run.bold) {
      target = appendElement(target, 'strong', '');
    }
    if (run.italic) {
      target = appendElement(target, 'em', '');
    }
    target.appendChild(parent.ownerDocument.createTextNode(run.text));
  });
}

function appendParagraph(parent, paragraph, className) {
  appendRuns(appendElement(parent, 'p', className || ''), paragraph.runs);
}

function appendCaption(parent, block) {
  const caption = appendElement(parent, 'figcaption', 'jats-caption');
  const heading = [block.label, block.title]
    .filter((part) => part && part !== block.label)
    .join('. ');
  if (block.label) {
    appendElement(caption, 'span', 'jats-label', block.label);
  }
  if (heading && heading !== block.label) {
    appendElement(caption, 'span', 'jats-caption-title', ' ' + block.title);
  }
  block.paragraphs.forEach((paragraph) => appendParagraph(caption, paragraph));
}

function appendFigure(parent, block) {
  const figure = appendElement(parent, 'figure', 'jats-figure');
  appendCaption(figure, block);
  if (block.graphicHref) {
    appendElement(
      figure, 'p', 'jats-graphic',
      'Image: ' + block.graphicHref + ' (in the zip output)'
    );
  }
}

function appendTable(parent, block) {
  const figure = appendElement(parent, 'figure', 'jats-table');
  appendCaption(figure, block);
  if (!block.rows.length) {
    return;
  }
  const table = appendElement(figure, 'table', '');
  const tbody = appendElement(table, 'tbody', '');
  block.rows.forEach((row) => {
    const tableRow = appendElement(tbody, 'tr', '');
    row.forEach((cell) => appendRuns(appendElement(tableRow, 'td', ''), cell.runs));
  });
}

function appendBlocks(parent, blocks, level) {
  blocks.forEach((block) => {
    if (block.type === 'paragraph') {
      appendParagraph(parent, block);
    } else if (block.type === 'figure') {
      appendFigure(parent, block);
    } else if (block.type === 'table') {
      appendTable(parent, block);
    } else if (block.type === 'section') {
      appendSection(parent, block, level);
    }
  });
}

function appendSection(parent, section, level) {
  const element = appendElement(parent, 'section', 'jats-section');
  if (section.title) {
    appendElement(element, 'h' + Math.min(level, 6), '', section.title);
  }
  appendBlocks(element, section.blocks, level + 1);
}

function appendAuthors(parent, viewModel) {
  if (!viewModel.authors.length) {
    return;
  }
  const affiliationNumberById = {};
  viewModel.affiliations.forEach((affiliation, index) => {
    affiliationNumberById[affiliation.id] = index + 1;
  });
  const list = appendElement(parent, 'p', 'jats-authors');
  viewModel.authors.forEach((author, index) => {
    const marks = author.affiliationIds
      .map((id) => affiliationNumberById[id])
      .filter((number) => number)
      .join(',');
    const text = author.name + (marks ? ' [' + marks + ']' : '');
    appendElement(list, 'span', 'jats-author', (index ? '; ' : '') + text);
  });
  if (!viewModel.affiliations.length) {
    return;
  }
  const affiliations = appendElement(parent, 'ol', 'jats-affiliations');
  viewModel.affiliations.forEach((affiliation) => {
    appendElement(affiliations, 'li', '', affiliation.text);
  });
}

function getReferenceText(reference) {
  const authors = reference.authors.join(', ');
  const pages = [reference.firstPage, reference.lastPage]
    .filter((page) => page)
    .join('-');
  const where = [
    reference.source,
    reference.volume && reference.issue
      ? reference.volume + '(' + reference.issue + ')'
      : reference.volume,
    pages,
    reference.year
  ].filter((part) => part).join(', ');
  return [authors, reference.articleTitle, where]
    .filter((part) => part)
    .join('. ');
}

function appendReferences(parent, references) {
  if (!references.length) {
    return;
  }
  appendElement(parent, 'h2', '', 'References');
  const list = appendElement(parent, 'ol', 'jats-references');
  references.forEach((reference) => {
    const item = appendElement(list, 'li', '', getReferenceText(reference));
    if (reference.doi) {
      appendElement(item, 'span', 'jats-doi', ' doi:' + reference.doi);
    }
  });
}

export function renderArticleViewModel(viewModel, document) {
  const article = document.createElement('article');
  article.className = 'jats-article';
  if (viewModel.journal) {
    appendElement(article, 'p', 'jats-journal', viewModel.journal);
  }
  appendElement(article, 'h1', '', viewModel.title || 'Untitled');
  appendAuthors(article, viewModel);
  viewModel.abstracts.forEach((abstract) => {
    const section = appendElement(article, 'section', 'jats-abstract');
    appendElement(section, 'h2', '', abstract.title || 'Abstract');
    abstract.paragraphs.forEach(
      (paragraph) => appendParagraph(section, paragraph)
    );
  });
  appendBlocks(article, viewModel.sections, 2);
  if (viewModel.acknowledgements.length) {
    const section = appendElement(article, 'section', 'jats-ack');
    appendElement(section, 'h2', '', 'Acknowledgements');
    appendBlocks(section, viewModel.acknowledgements, 3);
  }
  appendBlocks(article, viewModel.appendices, 2);
  appendReferences(article, viewModel.references);
  return article;
}

export function renderJatsXml(xmlText, document) {
  const xmlDocument = new DOMParser().parseFromString(xmlText, 'application/xml');
  if (xmlDocument.getElementsByTagName('parsererror').length) {
    throw new Error('the response is not well-formed XML');
  }
  return renderArticleViewModel(
    getArticleViewModel(getXmlNode(xmlDocument.documentElement)),
    document
  );
}
