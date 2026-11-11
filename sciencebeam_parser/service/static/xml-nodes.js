/*
 * Reading an XML document as plain objects, and saying where each part of it
 * came from.
 *
 * A node is `{name, attributes, children}`, children being nodes or strings.
 * Nothing here knows any vocabulary, so a reader of one lives beside a reader
 * of another and both say what they found in the same way.
 */

export function isElement(node) {
  return !!node && typeof node === 'object' && typeof node.name === 'string';
}

export function getChildElements(node, name) {
  if (!isElement(node)) {
    return [];
  }
  return (node.children || []).filter(
    (child) => isElement(child) && (!name || child.name === name)
  );
}

export function getFirstChildElement(node, name) {
  return getChildElements(node, name)[0] || null;
}

export function getDescendantElements(node, name) {
  const found = [];
  getChildElements(node).forEach((child) => {
    if (child.name === name) {
      found.push(child);
    }
    found.push(...getDescendantElements(child, name));
  });
  return found;
}

export function getFirstDescendantElement(node, name) {
  return getDescendantElements(node, name)[0] || null;
}

export function getAttribute(node, name) {
  return (isElement(node) && (node.attributes || {})[name]) || '';
}

/* The language an element declares, which `xml:lang` carries in both formats. */
export function getLanguage(node) {
  return getAttribute(node, 'xml:lang') || null;
}

export function getTextContent(node) {
  if (typeof node === 'string') {
    return node;
  }
  if (!isElement(node)) {
    return '';
  }
  return (node.children || []).map(getTextContent).join('');
}

export function getChildText(node, name) {
  const child = getFirstChildElement(node, name);
  return child ? getTextContent(child).trim() : '';
}

/* Where in the document a value was found.
 *
 * A name says what an element is and an id says which one it is, but a
 * document carries many a title and most of them carry no id. The path is
 * what separates them, and what finds the element again in the XML beside
 * the reading. It is assigned once over the tree rather than carried down
 * every function that reads one.
 */
export function addPaths(node, parentPath) {
  const children = getChildElements(node);
  const totalByName = {};
  const seenByName = {};
  children.forEach((child) => {
    totalByName[child.name] = (totalByName[child.name] || 0) + 1;
  });
  children.forEach((child) => {
    seenByName[child.name] = (seenByName[child.name] || 0) + 1;
    const step = totalByName[child.name] > 1
      ? `${child.name}[${seenByName[child.name]}]`
      : child.name;
    child.path = `${parentPath}/${step}`;
    addPaths(child, child.path);
  });
}


/* Every child of that name as a value, each carrying the language it declares. */
export function getValuesWithLanguage(node, name, label) {
  return getChildElements(node, name)
    .map((child) => {
      const text = getTextContent(child).trim();
      if (!text) {
        return null;
      }
      const language = getLanguage(child);
      const value = { label: label || name, text, path: child.path };
      return language ? { ...value, language } : value;
    })
    .filter((value) => value !== null);
}


export function getValue(node, name, label) {
  const child = getFirstChildElement(node, name);
  const text = child ? getTextContent(child).trim() : '';
  return text ? { label: label || name, text, path: child.path } : null;
}

/* The runs of text a passage is made of, as the caller's vocabulary has them.
 *
 * Which element means bold and which points at a reference is the one thing
 * about a passage that is not the same in two formats, so the caller says.
 * `getInline` is given an element and answers `{style}` to go on inside it
 * with that added, `{leaf}` to take the whole of it as one run, or nothing to
 * go on inside it unchanged.
 */
export function getInlineRuns(node, getInline, style) {
  const currentStyle = style || {};
  const runs = [];
  (isElement(node) ? node.children || [] : []).forEach((child) => {
    if (typeof child === 'string') {
      runs.push({ ...currentStyle, text: child });
      return;
    }
    const inline = getInline(child) || {};
    if (inline.leaf) {
      runs.push({
        ...currentStyle, ...inline.leaf, text: getTextContent(child)
      });
      return;
    }
    runs.push(...getInlineRuns(
      child, getInline, { ...currentStyle, ...(inline.style || {}) }
    ));
  });
  return runs.filter((run) => run.text !== '');
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


export function parseXmlNode(xmlText) {
  const xmlDocument = new DOMParser().parseFromString(xmlText, 'application/xml');
  if (xmlDocument.getElementsByTagName('parsererror').length) {
    throw new Error('the response is not well-formed XML');
  }
  const node = getXmlNode(xmlDocument.documentElement);
  node.path = `/${node.name}`;
  addPaths(node, node.path);
  return node;
}
