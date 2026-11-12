/*
 * Drawing an article that has already been read, whatever it was read from.
 *
 * Nothing here knows JATS or TEI. It is given a view model -- values that say
 * what they are called, which one they are and where they came from -- and
 * draws it, so a second format is a second reader rather than a second page.
 *
 * `renderArticleViewModel` takes an optional `{getAssetUrl}`, which is asked
 * for a figure's graphic by the name the document gives it. Where the figures
 * travel beside the XML, as they do in the archive the service answers with,
 * that is what puts them on the page. `getGraphicReferences` names what a
 * reading will ask for and which figure asks for it, so a caller holding an
 * archive can fetch those, leave the rest of it alone, and say what the rest
 * of it was for.
 */

function appendTag(element, marked) {
  const tag = element.ownerDocument.createElement('span');
  tag.className = 'jats-tag';
  tag.textContent = marked.label;
  if (marked.path) {
    tag.setAttribute('title', marked.path);
  }
  if (marked.id) {
    // The id is what a cross-reference points at, so the thing it points at
    // says which one it is.
    const tagId = element.ownerDocument.createElement('span');
    tagId.className = 'jats-tag-id';
    tagId.textContent = '#' + marked.id;
    tag.appendChild(tagId);
  }
  element.appendChild(tag);
}

function appendElement(parent, tagName, className, marked) {
  const element = parent.ownerDocument.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (marked && marked.label) {
    element.setAttribute('data-jats', marked.label);
    appendTag(element, marked);
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
    parent, tagName || 'span', className || '', value
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
    appendElement(parent, 'p', className || '', paragraph),
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
  const element = appendElement(parent, 'div', 'jats-graphic', graphic);
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
  const figure = appendElement(parent, 'figure', 'jats-figure', block);
  appendCaption(figure, block);
  block.graphics.forEach((graphic) => appendGraphic(figure, graphic, options));
}

function appendTable(parent, block, options) {
  const figure = appendElement(parent, 'figure', 'jats-table', block);
  appendCaption(figure, block);
  (block.graphics || []).forEach((graphic) => appendGraphic(figure, graphic, options));
  if (!block.rows.length) {
    return;
  }
  const table = appendElement(figure, 'table', '');
  const tbody = appendElement(table, 'tbody', '');
  block.rows.forEach((row) => {
    const tableRow = appendElement(tbody, 'tr', '');
    row.forEach((cell) => appendRuns(
      appendElement(tableRow, 'td', '', cell), cell.runs
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
      appendTable(parent, block, options);
    } else if (block.type === 'section') {
      appendSection(parent, block, level, options);
    }
  });
}

function appendSection(parent, section, level, options) {
  const element = appendElement(parent, 'section', 'jats-section', section);
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
 * reads as the number the list gives that affiliation, its name says which id
 * it points at the way any other id is said, it carries the affiliation's own
 * text to rest on, and following it goes to the entry rather than leaving the
 * reader to count.
 */
function appendAffiliationRef(parent, ref, affiliationByRid) {
  const affiliation = affiliationByRid[ref.rid];
  const anchor = appendElement(
    parent, 'a', 'jats-xref', { ...ref, id: ref.rid }
  );
  anchor.setAttribute('href', '#' + ref.rid);
  anchor.setAttribute(
    'title', affiliation ? affiliation.text : 'no affiliation with id ' + ref.rid
  );
  appendText(anchor, affiliation ? String(affiliation.number) : ref.rid);
}

function appendAuthor(parent, author, affiliationByRid) {
  const element = appendElement(parent, 'span', 'jats-author', author);
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
  const list = appendElement(
    parent, 'p', 'jats-authors', viewModel.authorGroup
  );
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
    const item = appendElement(affiliations, 'li', '', affiliation);
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
  const item = appendElement(parent, 'li', '', reference);
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

function appendTitle(parent, value, tagName) {
  const title = appendValue(parent, value, tagName, 'jats-title');
  if (title && value.language) {
    // On the element as well as beside it, so a reader hears the right voice
    // and a reader of the page can see which title they were given.
    title.setAttribute('lang', value.language);
    const language = appendElement(title, 'span', 'jats-language');
    language.textContent = value.language;
  }
}


export function renderArticleViewModel(viewModel, document, options) {
  const renderOptions = options || {};
  const article = document.createElement('article');
  article.className = 'jats-article';
  appendValue(article, viewModel.journal, 'p', 'jats-journal');
  if (viewModel.titles.length) {
    viewModel.titles.forEach(
      (value, index) => appendTitle(article, value, index === 0 ? 'h1' : 'h2')
    );
  } else {
    appendElement(article, 'h1', '').textContent = 'Untitled';
  }
  appendAuthors(article, viewModel);
  viewModel.abstracts.forEach((abstract) => {
    const section = appendElement(
      article, 'section', 'jats-abstract', abstract
    );
    const heading = abstract.title
      ? appendValue(section, abstract.title, 'h2')
      : appendElement(section, 'h2', '');
    if (!abstract.title) {
      heading.textContent = 'Abstract';
    }
    if (abstract.language) {
      section.setAttribute('lang', abstract.language);
      appendElement(heading, 'span', 'jats-language')
        .textContent = abstract.language;
    }
    abstract.paragraphs.forEach(
      (paragraph) => appendParagraph(section, paragraph)
    );
  });
  appendBlocks(article, viewModel.sections, 2, renderOptions);
  if (viewModel.acknowledgements.length) {
    const section = appendElement(
      article, 'section', 'jats-ack', { label: 'ack' }
    );
    appendElement(section, 'h2', '').textContent = 'Acknowledgements';
    appendBlocks(section, viewModel.acknowledgements, 3, renderOptions);
  }
  appendBlocks(article, viewModel.appendices, 2, renderOptions);
  appendReferences(article, viewModel.references);
  return article;
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
    if (block.type === 'figure' || block.type === 'table') {
      (block.graphics || []).forEach((graphic) => {
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
