/*
 * Turning JATS into something a person can read, without hiding what it read.
 *
 * Nothing here imports anything or assumes the page around it, so that a site
 * other than this one can use the same reading of an article.
 *
 * `getArticleViewModel` is the part that knows JATS. It takes plain objects --
 * `{name, attributes, children}`, children being nodes or strings -- rather
 * than DOM nodes, so what an article reads like can be tested without a
 * browser, and so a consumer rendering with something other than the DOM gets
 * the same reading.
 *
 * Every piece of text the model carries states the element it came from, and
 * the smallest labelled piece is its own element in the output, carrying that
 * name as `data-jats`. Reading an extraction is most of the time asking
 * whether a value was labelled as the right thing, and a rendering that
 * answers only "what does it say" cannot be checked.
 *
 * `renderArticleViewModel` takes an optional `{getAssetUrl}`, which is asked
 * for a figure's graphic by the name the document gives it. Where the figures
 * travel beside the XML, as they do in the archive the service answers with,
 * that is what puts them on the page. `getGraphicReferences` names what a
 * reading will ask for and which figure asks for it, so a caller holding an
 * archive can fetch those, leave the rest of it alone, and say what the rest
 * of it was for.
 */

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

function getValue(node, name, label) {
  const text = getChildText(node, name);
  return text ? { label: label || name, text } : null;
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
  return { label: node.name, runs: getInlineRuns(node) };
}

function getCaption(node) {
  const caption = getFirstChildElement(node, 'caption');
  return {
    label: node.name,
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
    id: getAttribute(node, 'id'),
    title: getValue(node, 'title'),
    blocks: getSectionBlocks(node)
  };
}

function getPersonNameParts(node) {
  const name = getFirstChildElement(node, 'name');
  if (!name) {
    const text = getTextContent(node).trim();
    return text ? [{ label: node.name, text }] : [];
  }
  return [
    getValue(name, 'given-names'),
    getValue(name, 'surname')
  ].filter((part) => part);
}

function getAuthors(articleMeta) {
  const contribGroup = getFirstChildElement(articleMeta, 'contrib-group');
  return getChildElements(contribGroup, 'contrib').map((contrib) => ({
    label: contrib.name,
    nameParts: getPersonNameParts(contrib),
    email: getValue(contrib, 'email'),
    affiliationRefs: getChildElements(contrib, 'xref')
      .filter((xref) => getAttribute(xref, 'ref-type') === 'aff')
      .map((xref) => ({ label: xref.name, rid: getAttribute(xref, 'rid') }))
      .filter((ref) => ref.rid)
  }));
}

function getAffiliations(articleMeta) {
  return getChildElements(articleMeta, 'aff').map((aff) => ({
    label: aff.name,
    id: getAttribute(aff, 'id'),
    parts: AFFILIATION_PARTS.flatMap(([name, contentType]) => getChildElements(aff, name)
      .filter(
        (child) => !contentType
          || getAttribute(child, 'content-type') === contentType
      )
      .map((child) => ({
        label: contentType || name,
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
    id: getAttribute(node, 'id'),
    authors: getChildElements(personGroup, 'name').map((name) => [
      getValue(name, 'given-names'),
      getValue(name, 'surname')
    ].filter((part) => part)),
    doi: doi ? { label: 'pub-id', text: getTextContent(doi).trim() } : null
  };
  REFERENCE_FIELDS.forEach(([key, name]) => {
    reference[key] = citation ? getValue(citation, name) : null;
  });
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
  const journalTitle = front
    ? getFirstDescendantElement(front, 'journal-title')
    : null;
  return {
    title: titleGroup ? getValue(titleGroup, 'article-title') : null,
    journal: journalTitle
      ? { label: 'journal-title', text: getTextContent(journalTitle).trim() }
      : null,
    authors: getAuthors(articleMeta),
    affiliations: getAffiliations(articleMeta),
    abstracts: getChildElements(articleMeta, 'abstract').map((abstract) => ({
      label: abstract.name,
      title: getValue(abstract, 'title'),
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

function appendElement(parent, tagName, className, label) {
  const element = parent.ownerDocument.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (label) {
    element.setAttribute('data-jats', label);
    const tag = parent.ownerDocument.createElement('span');
    tag.className = 'jats-tag';
    tag.textContent = label;
    element.appendChild(tag);
  }
  parent.appendChild(element);
  return element;
}

function appendText(parent, text) {
  parent.appendChild(parent.ownerDocument.createTextNode(text));
}

function appendValue(parent, value, tagName, className) {
  if (!value) {
    return null;
  }
  const element = appendElement(
    parent, tagName || 'span', className || '', value.label
  );
  appendText(element, value.text);
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
    appendText(target, run.text);
  });
}

function appendParagraph(parent, paragraph, className) {
  appendRuns(
    appendElement(parent, 'p', className || '', paragraph.label),
    paragraph.runs
  );
}

function appendCaption(parent, block) {
  const caption = appendElement(parent, 'figcaption', 'jats-caption');
  appendValue(caption, block.heading, 'span', 'jats-label');
  if (block.title && (!block.heading || block.title.text !== block.heading.text)) {
    appendText(caption, ' ');
    appendValue(caption, block.title, 'span', 'jats-caption-title');
  }
  block.paragraphs.forEach((paragraph) => appendParagraph(caption, paragraph));
}

function appendGraphic(parent, graphic, options) {
  const element = appendElement(parent, 'div', 'jats-graphic', graphic.label);
  if (!graphic.href) {
    appendElement(element, 'span', 'jats-graphic-name').textContent = (
      'no file named for this graphic'
    );
    return;
  }
  const url = options.getAssetUrl ? options.getAssetUrl(graphic.href) : null;
  if (!url) {
    appendText(element, graphic.href + ' (in the zip output)');
    return;
  }
  const image = appendElement(element, 'img', 'jats-graphic-image');
  image.setAttribute('src', url);
  image.setAttribute('alt', graphic.href);
  appendElement(element, 'span', 'jats-graphic-name').textContent = graphic.href;
}

function appendFigure(parent, block, options) {
  const figure = appendElement(parent, 'figure', 'jats-figure', block.label);
  appendCaption(figure, block);
  block.graphics.forEach((graphic) => appendGraphic(figure, graphic, options));
}

function appendTable(parent, block) {
  const figure = appendElement(parent, 'figure', 'jats-table', block.label);
  appendCaption(figure, block);
  if (!block.rows.length) {
    return;
  }
  const table = appendElement(figure, 'table', '');
  const tbody = appendElement(table, 'tbody', '');
  block.rows.forEach((row) => {
    const tableRow = appendElement(tbody, 'tr', '');
    row.forEach((cell) => appendRuns(
      appendElement(tableRow, 'td', '', cell.label), cell.runs
    ));
  });
}

function appendBlocks(parent, blocks, level, options) {
  blocks.forEach((block) => {
    if (block.type === 'paragraph') {
      appendParagraph(parent, block);
    } else if (block.type === 'figure') {
      appendFigure(parent, block, options);
    } else if (block.type === 'table') {
      appendTable(parent, block);
    } else if (block.type === 'section') {
      appendSection(parent, block, level, options);
    }
  });
}

function appendSection(parent, section, level, options) {
  const element = appendElement(parent, 'section', 'jats-section', section.label);
  appendValue(element, section.title, 'h' + Math.min(level, 6));
  appendBlocks(element, section.blocks, level + 1, options);
}

function getAffiliationByRid(affiliations) {
  const byRid = {};
  affiliations.forEach((affiliation, index) => {
    byRid[affiliation.id] = {
      number: index + 1,
      text: affiliation.parts.map((part) => part.text).join(', ')
    };
  });
  return byRid;
}

/* The reference an author makes to an affiliation, as itself.
 *
 * A number beside a name says there is a link without saying to what. The
 * reference is an element the document carries, so it is one here too: it
 * reads as the number the list gives that affiliation, it says which id it
 * points at, it carries the affiliation's own text to rest on, and following
 * it goes to the entry rather than leaving the reader to count.
 */
function appendAffiliationRef(parent, ref, affiliationByRid) {
  const affiliation = affiliationByRid[ref.rid];
  const anchor = appendElement(parent, 'a', 'jats-xref', ref.label);
  anchor.setAttribute('href', '#' + ref.rid);
  anchor.setAttribute(
    'title', affiliation ? affiliation.text : 'no affiliation with id ' + ref.rid
  );
  appendText(anchor, affiliation ? String(affiliation.number) : ref.rid);
}

function appendAuthor(parent, author, affiliationByRid) {
  const element = appendElement(parent, 'span', 'jats-author', author.label);
  appendAuthorContent(element, author, affiliationByRid);
  return element;
}

function appendAuthorContent(element, author, affiliationByRid) {
  author.nameParts.forEach((part, index) => {
    if (index) {
      appendText(element, ' ');
    }
    appendValue(element, part);
  });
  if (author.email) {
    appendText(element, ' ');
    appendValue(element, author.email);
  }
  author.affiliationRefs.forEach((ref) => {
    appendText(element, ' ');
    appendAffiliationRef(element, ref, affiliationByRid);
  });
}

/* What separates one author from the next is the page's, not the document's.
 *
 * A contrib ends where the element ends; the comma is there so a line of
 * names reads as one. It sits inside the author it follows rather than
 * between two of them, so that showing each author on its own line does not
 * leave it stranded on a line of its own. */
function appendAuthorSeparator(element) {
  appendElement(element, 'span', 'jats-separator').textContent = ', ';
}

function appendAuthors(parent, viewModel) {
  if (!viewModel.authors.length) {
    return;
  }
  const affiliationByRid = getAffiliationByRid(viewModel.affiliations);
  const list = appendElement(parent, 'p', 'jats-authors');
  viewModel.authors.forEach((author, index) => {
    const element = appendAuthor(list, author, affiliationByRid);
    if (index < viewModel.authors.length - 1) {
      appendAuthorSeparator(element);
    }
  });
  if (!viewModel.affiliations.length) {
    return;
  }
  const affiliations = appendElement(parent, 'ol', 'jats-affiliations');
  viewModel.affiliations.forEach((affiliation) => {
    const item = appendElement(affiliations, 'li', '', affiliation.label);
    if (affiliation.id) {
      item.setAttribute('id', affiliation.id);
    }
    affiliation.parts.forEach((part, index) => {
      if (index) {
        appendText(item, ', ');
      }
      appendValue(item, part);
    });
  });
}

function appendReferenceAuthors(parent, authors) {
  authors.forEach((nameParts, index) => {
    if (index) {
      appendText(parent, ', ');
    }
    nameParts.forEach((part, partIndex) => {
      if (partIndex) {
        appendText(parent, ' ');
      }
      appendValue(parent, part);
    });
  });
}

function appendReferenceVolume(parent, reference) {
  appendValue(parent, reference.volume);
  if (!reference.issue) {
    return;
  }
  if (reference.volume) {
    appendText(parent, '(');
  }
  appendValue(parent, reference.issue);
  if (reference.volume) {
    appendText(parent, ')');
  }
}

function appendReferencePages(parent, reference) {
  appendValue(parent, reference.firstPage);
  if (reference.firstPage && reference.lastPage) {
    appendText(parent, '-');
  }
  appendValue(parent, reference.lastPage);
}

function getReferenceWhereParts(reference) {
  const parts = [];
  if (reference.source) {
    parts.push((target) => appendValue(target, reference.source));
  }
  if (reference.volume || reference.issue) {
    parts.push((target) => appendReferenceVolume(target, reference));
  }
  if (reference.firstPage || reference.lastPage) {
    parts.push((target) => appendReferencePages(target, reference));
  }
  if (reference.year) {
    parts.push((target) => appendValue(target, reference.year));
  }
  return parts;
}

function appendJoined(parent, parts, separator) {
  parts.forEach((append, index) => {
    if (index) {
      appendText(parent, separator);
    }
    append(parent);
  });
}

function appendReference(parent, reference) {
  const item = appendElement(parent, 'li', '', reference.label);
  const groups = [];
  if (reference.authors.length) {
    groups.push((target) => appendReferenceAuthors(target, reference.authors));
  }
  if (reference.articleTitle) {
    groups.push((target) => appendValue(target, reference.articleTitle));
  }
  const whereParts = getReferenceWhereParts(reference);
  if (whereParts.length) {
    groups.push((target) => appendJoined(target, whereParts, ', '));
  }
  appendJoined(item, groups, '. ');
  if (reference.doi) {
    appendText(item, groups.length ? ' doi:' : 'doi:');
    appendValue(item, reference.doi);
  }
}

function appendReferences(parent, references) {
  if (!references.length) {
    return;
  }
  appendElement(parent, 'h2', '').textContent = 'References';
  const list = appendElement(parent, 'ol', 'jats-references');
  references.forEach((reference) => appendReference(list, reference));
}

export function renderArticleViewModel(viewModel, document, options) {
  const renderOptions = options || {};
  const article = document.createElement('article');
  article.className = 'jats-article';
  appendValue(article, viewModel.journal, 'p', 'jats-journal');
  if (viewModel.title) {
    appendValue(article, viewModel.title, 'h1');
  } else {
    appendElement(article, 'h1', '').textContent = 'Untitled';
  }
  appendAuthors(article, viewModel);
  viewModel.abstracts.forEach((abstract) => {
    const section = appendElement(
      article, 'section', 'jats-abstract', abstract.label
    );
    if (abstract.title) {
      appendValue(section, abstract.title, 'h2');
    } else {
      appendElement(section, 'h2', '').textContent = 'Abstract';
    }
    abstract.paragraphs.forEach(
      (paragraph) => appendParagraph(section, paragraph)
    );
  });
  appendBlocks(article, viewModel.sections, 2, renderOptions);
  if (viewModel.acknowledgements.length) {
    const section = appendElement(article, 'section', 'jats-ack', 'ack');
    appendElement(section, 'h2', '').textContent = 'Acknowledgements';
    appendBlocks(section, viewModel.acknowledgements, 3, renderOptions);
  }
  appendBlocks(article, viewModel.appendices, 2, renderOptions);
  appendReferences(article, viewModel.references);
  return article;
}

export function parseJatsViewModel(xmlText) {
  const xmlDocument = new DOMParser().parseFromString(xmlText, 'application/xml');
  if (xmlDocument.getElementsByTagName('parsererror').length) {
    throw new Error('the response is not well-formed XML');
  }
  return getArticleViewModel(getXmlNode(xmlDocument.documentElement));
}

function getFigureName(figure) {
  if (figure.heading && figure.heading.text) {
    return figure.heading.text;
  }
  if (figure.title && figure.title.text) {
    return figure.title.text;
  }
  return figure.id || 'a figure';
}

function addGraphicReferences(references, blocks) {
  blocks.forEach((block) => {
    if (block.type === 'figure') {
      block.graphics.forEach((graphic) => {
        if (graphic.href) {
          references.push({ href: graphic.href, figure: getFigureName(block) });
        }
      });
    } else if (block.type === 'section') {
      addGraphicReferences(references, block.blocks);
    }
  });
}

export function getGraphicReferences(viewModel) {
  const references = [];
  addGraphicReferences(references, viewModel.sections);
  addGraphicReferences(references, viewModel.acknowledgements);
  addGraphicReferences(references, viewModel.appendices);
  return references;
}

export function getGraphicHrefs(viewModel) {
  return Array.from(new Set(
    getGraphicReferences(viewModel).map((reference) => reference.href)
  ));
}

export function renderJatsXml(xmlText, document, options) {
  return renderArticleViewModel(
    parseJatsViewModel(xmlText), document, options
  );
}
