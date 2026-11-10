/*
 * Enough of a document for the renderer to build into, and a way to read back
 * what it built.
 *
 * The renderer wants a document to create elements with, and node has none.
 * What it uses of one is small and listed here: enough to check that what a
 * view model knows reaches the markup, not a stand-in for a browser.
 */
function createTextNode(text) {
  return { nodeType: 3, text };
}

function createElement(ownerDocument, tagName) {
  return {
    nodeType: 1,
    tagName,
    ownerDocument,
    className: '',
    attributes: {},
    childNodes: [],
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
    appendChild(child) {
      this.childNodes.push(child);
      return child;
    },
    set textContent(value) {
      this.childNodes = [createTextNode(value)];
    }
  };
}

function createDocument() {
  const document = {};
  document.createElement = (tagName) => createElement(document, tagName);
  document.createTextNode = createTextNode;
  return document;
}

function escapeText(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function serialize(node) {
  if (node.nodeType === 3) {
    return escapeText(node.text);
  }
  const attributes = { ...node.attributes };
  if (node.className) {
    attributes.class = node.className;
  }
  const attributeText = Object.entries(attributes)
    .map(([name, value]) => ` ${name}="${escapeText(value)}"`)
    .join('');
  return (
    `<${node.tagName}${attributeText}>`
    + node.childNodes.map(serialize).join('')
    + `</${node.tagName}>`
  );
}

export { createDocument, serialize };
