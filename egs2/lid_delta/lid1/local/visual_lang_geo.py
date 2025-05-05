# To run this script, `pip install -U kaleido` is required.

import os
import configparser
import pandas as pd
import plotly.express as px
import plotly.io as pio
import argparse

GLOTTOLOG = "/work/nvme/bbjs/qwang20/glottolog/languoids/tree"

pio.templates.default = "plotly_white"

def extract_languages_by_iso(glottolog_tree_dir, target_iso_codes):
    results = []
    for root, dirs, files in os.walk(glottolog_tree_dir):
        if 'md.ini' in files:
            path = os.path.join(root, 'md.ini')
            config = configparser.ConfigParser()
            config.read(path, encoding='utf-8')

            try:
                iso = config.get('core', 'iso639-3')
                if iso in target_iso_codes:
                    name = config.get('core', 'name')
                    latitude = config.getfloat('core', 'latitude')
                    longitude = config.getfloat('core', 'longitude')
                    results.append({
                        'iso639_3': iso,
                        'name': name,
                        'latitude': latitude,
                        'longitude': longitude
                    })
            except (configparser.NoOptionError, configparser.NoSectionError, ValueError):
                continue
    return pd.DataFrame(results)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize language locations")
    parser.add_argument(
        "--iso_codes",
        type=str,
        nargs='+',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output files",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    iso_codes = args.iso_codes
    output_html = os.path.join(args.output_dir, "language_map.html")
    output_png = os.path.join(args.output_dir, "language_map.png")

    df = extract_languages_by_iso(GLOTTOLOG, iso_codes)

    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        hover_name='name',
        text='iso639_3',
        projection='natural earth',
        title='Language Locations by ISO 639-3 Code',
    )

    fig.update_traces(
        marker=dict(size=10, color='crimson', line=dict(width=1, color='black')),
        textposition='top center',
        textfont=dict(size=12, color='black')
    )
    fig.update_layout(
        title_font_size=22,
        font=dict(family="Arial", size=14),
        geo=dict(
            showland=True,
            landcolor="rgb(230, 230, 230)",
            showcountries=True,
            countrycolor="rgb(180, 180, 180)",
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.write_html(output_html)
    fig.write_image(output_png, width=1400, height=800, scale=2)

    print(f"✅ Saved interactive map to {output_html}")
    print(f"✅ Saved high-res PNG image to {output_png}")
