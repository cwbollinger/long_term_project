#! /usr/bin/env python
import yaml
import os
import sys

import rosbag
from tf_bag import BagTfTransformer

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


def get_map_as_im(mapfile):

    directory = os.path.dirname(mapfile)

    with open(mapfile, 'r') as stream:
        map_metadata = yaml.safe_load(stream)

    map_data = plt.imread(os.path.join(directory, map_metadata['image']))
    dims = map_data.shape
    height = dims[0] * map_metadata['resolution']
    width = dims[1] * map_metadata['resolution']
    x_origin = map_metadata['origin'][0]
    y_origin = map_metadata['origin'][1]
    map_extent = [x_origin, width+x_origin, y_origin, height+y_origin]
    return map_data, map_extent


def find_unvisualized_logs():
    bagdir = os.path.expanduser('~/bags')
    visdir = os.path.expanduser('~/bags/visualization')
    mapfile = os.path.expanduser('~/maps/graf.yaml')

    files = os.listdir(bagdir)
    visfiles = os.listdir(visdir)

    for f in files:
        parts = f.split('_')
        if parts[-1][:4] != 'wifi':
            continue
        visname = '{}_{}_{}_wifi_visualization.png'.format(parts[0], parts[1], parts[2])
        if visname not in visfiles:
            fullfile = os.path.join(bagdir, f)
            generate_wifi_map(mapfile, fullfile, visname)


def generate_wifi_map(mapname, bagname, outputname):
    graf_map, extent = get_map_as_im(mapname)

    bag = rosbag.Bag(bagname)
    bag_transformer = BagTfTransformer(bag)
    #print(bag_transformer.getTransformGraphInfo())
    #print(bag_transformer.getChain('map', 'base_link'))

    output = bag_transformer.lookupTransformWhenTransformUpdates(
        'map',
        'base_link',
        trigger_orig_frame='map',
        trigger_dest_frame='odom')

    wifi_msgs = []
    for topic, msg, t in bag.read_messages():
        if topic.strip('/') not in ('tf', 'tf_static'):
            wifi_msgs.append((t, msg))

    fig, ax = plt.subplots()
    # ax.imshow(graf_map, extent=extent, cmap=plt.get_cmap('Greys_r'))
    cmap = sns.cubehelix_palette(dark=1, light=0, as_cmap=True) 
    ax.imshow(graf_map, extent=extent, cmap=cmap)
    points = []
    idx = 0
    unique_wifi_vals = set()
    for t, trans in output:
        if wifi_msgs[idx][0] < t:
            pos, q = trans
            quality = wifi_msgs[idx][1].quality
            unique_wifi_vals.add(quality)
            points.append({'x': pos[0], 'y': pos[1], 'wifi': quality})
            idx += 1

        if idx >= len(wifi_msgs):
            break

    # colors = sns.color_palette('muted', n_colors=3)
    # sns.scatterplot(
    #     x='x',
    #     y='y',
    #     hue='wifi',
    #     legend=False,
    #     data=pd.DataFrame(points),
    #     ax=ax,
    #     linewidth=0,
    #     palette=sns.color_palette("RdYlGn", n_colors=len(unique_wifi_vals)))

    sns.scatterplot(
        x='x',
        y='y',
        hue='wifi',
        legend='brief',
        data=pd.DataFrame(points),
        ax=ax,
        linewidth=0,
        palette="Reds_r")

    plt.savefig(os.path.expanduser('~/bags/visualization/{}'.format(outputname)), bbox_inches='tight')


def main():
    while True:
        find_unvisualized_logs()
        time.sleep(1)


if __name__ == '__main__':
    main()
    # generate_wifi_map(
    #     os.path.expanduser('~/maps/graf.yaml'),
    #     sys.argv[1],
    #     'testmap.png'
    # )

