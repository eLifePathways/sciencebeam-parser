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

export function getValue(node, name, label) {
  const child = getFirstChildElement(node, name);
  const text = child ? getTextContent(child).trim() : '';
  return text ? { label: label || name, text, path: child.path } : null;
}

export function getInlineRuns(node, style) {
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
