/*
 * Runs the reading view's JATS side over a node tree on stdin.
 *
 * The page's own renderer needs a browser; what an article reads like does
 * not, and this is what lets the Python tests cover it.
 */
import {
  getArticleViewModel,
  getGraphicHrefs
} from '../../../sciencebeam_parser/service/static/jats-to-html.js';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const viewModel = getArticleViewModel(JSON.parse(input));
  process.stdout.write(JSON.stringify({
    ...viewModel,
    graphicHrefs: getGraphicHrefs(viewModel)
  }));
});
