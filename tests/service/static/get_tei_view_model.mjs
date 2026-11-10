/*
 * Runs the reading view's TEI side over a node tree on stdin.
 */
import {
  getGraphicHrefs,
  getGraphicReferences
} from '../../../sciencebeam_parser/service/static/article-view.js';
import {
  getTeiArticleViewModel
} from '../../../sciencebeam_parser/service/static/tei-to-html.js';

let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  const viewModel = getTeiArticleViewModel(JSON.parse(input));
  process.stdout.write(JSON.stringify({
    ...viewModel,
    graphicHrefs: getGraphicHrefs(viewModel),
    graphicReferences: getGraphicReferences(viewModel)
  }));
});
