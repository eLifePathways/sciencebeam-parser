/*
 * Renders the reading view over a node tree on stdin and writes the markup.
 *
 * The renderer wants a document to create elements with, and node has none.
 * What it uses of one is small and listed here: enough to check that what the
 * view model knows reaches the markup, not a stand-in for a browser.
 */
import {
  renderArticleViewModel
} from '../../../sciencebeam_parser/service/static/article-view.js';
import {
  getArticleViewModel
} from '../../../sciencebeam_parser/service/static/jats-to-html.js';

import { createDocument, serialize } from './stand_in_document.mjs';

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
