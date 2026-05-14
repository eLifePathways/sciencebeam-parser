#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sciencebeam_parser.utils.model_data_diff import format_model_data_diff  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description='Compare model feature data between GROBID and ScienceBeam Parser'
    )
    parser.add_argument('--sbeam', required=True, help='ScienceBeam Parser data file (.data)')
    parser.add_argument('--grobid', required=True, help='GROBID data file (.data)')
    parser.add_argument(
        '--feature-names',
        required=True,
        help='Feature names JSON sidecar saved from /api/models/{model}/feature-names'
    )
    args = parser.parse_args()

    feature_names = json.loads(
        Path(args.feature_names).read_text(encoding='utf-8')
    )['feature_names']
    sbeam_text = Path(args.sbeam).read_text(encoding='utf-8')
    grobid_text = Path(args.grobid).read_text(encoding='utf-8')

    print(format_model_data_diff(sbeam_text, grobid_text, feature_names))


if __name__ == '__main__':
    main()
