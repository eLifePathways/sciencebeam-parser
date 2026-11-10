/* Renders a TEI node tree the way the page does, over a stand-in document. */
import {
  renderArticleViewModel
} from '../../../sciencebeam_parser/service/static/article-view.js';
import {
  getTeiArticleViewModel
} from '../../../sciencebeam_parser/service/static/tei-to-html.js';
import { createDocument, serialize } from './stand_in_document.mjs';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const { node, assets } = JSON.parse(input);
  process.stdout.write(serialize(renderArticleViewModel(
    getTeiArticleViewModel(node),
    createDocument(),
    { getAssetUrl: (href) => (assets || {})[href] || null }
  )));
});
