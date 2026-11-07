/*
 * Renders the reading view over a node tree on stdin and writes the markup.
 *
 * The renderer wants a document to create elements with, and node has none.
 * What it uses of one is small and listed here: enough to check that what the
 * view model knows reaches the markup, not a stand-in for a browser.
 */
import {
  getArticleViewModel,
  renderArticleViewModel
} from '../../../sciencebeam_parser/service/static/jats-to-html.js';

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

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const { node, assets } = JSON.parse(input);
  const viewModel = getArticleViewModel(node);
  process.stdout.write(serialize(renderArticleViewModel(
    viewModel,
    createDocument(),
    { getAssetUrl: (href) => (assets || {})[href] || null }
  )));
});
