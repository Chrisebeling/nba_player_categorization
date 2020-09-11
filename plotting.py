import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.style as pstyle
from mpl_toolkits import mplot3d
import matplotlib.ticker as mtick
import seaborn as sns
import pickle

DESIRED_STATS = ['pts', 'trb', 'ast', 'blk', 'stl','tov','fg3_pct', 'fg2_pct', 'ft_pct', 'fg3a', 'fg2a', 'fta']
INFO_STATS = ['last_name','first_name','season','min_season', 'max_season']

def import_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        [stats, final_clusters, final_closest, final_cost, close_min_max] = pickle.load(f)

    # add fg2 stats
    stats.loc[:,'fg2a'] = stats.loc[:,'fga'] - stats.loc[:,'fg3a']
    stats.loc[:,'fg2'] = stats.loc[:,'fg'] - stats.loc[:,'fg3']
    stats.loc[:,'fg2_pct'] = stats.loc[:,'fg2'] / stats.loc[:,'fg2a']

    # 1 player has no fta, 20 have no 3pa
    for column in ['ft_pct', 'fg2_pct', 'fg3_pct']:
        stats.loc[:,column] = stats.loc[:,column].fillna(stats.loc[:,column].mean())

    
    info_stats = [x for x in INFO_STATS if x in stats.columns]
    data = stats[info_stats+DESIRED_STATS]
    X = data[DESIRED_STATS].to_numpy()
    
    return stats, final_clusters, final_closest, final_cost, close_min_max, data, X

def friendly_data(data, final_clusters, final_closest, close_min_max, desc_csv=None):
    final_data = data.copy()
    final_data.loc[:,'cluster'] = list(final_closest)
    final_data.loc[:,'first_last'] = final_data.loc[:,'first_name'].fillna('') +' '+final_data.loc[:,'last_name'].fillna('')
    if 'season' in final_data.columns:
        final_data.loc[:,'display_name'] = final_data.loc[:,'first_last'] + ' (' + final_data.loc[:,'season'].astype('str') + ')'
    else:
        final_data.loc[:,'display_name'] = final_data.loc[:,'first_last']
    
    cluster_summary = pd.DataFrame(final_clusters, columns = DESIRED_STATS)
    for header, idx in zip(['Closest', 'Min', 'Max'], range(3)):
        cluster_summary.loc[:,header] = list(final_data.loc[close_min_max[:,idx],'display_name'])
    close_players = np.resize(np.array(final_data.loc[close_min_max.flatten(), 'display_name']), (len(cluster_summary), 6))
    
    if desc_csv:
        cluster_summary = cluster_summary.merge(pd.read_csv(desc_csv, index_col=0), how='left', left_index=True, right_index=True)
        final_data = final_data.merge(cluster_summary[['cluster_name','cluster_rank']], how='left', left_on='cluster',right_index=True)
    
    return final_data, cluster_summary

def plot_career(player, final_data, cluster_summary):
    players = [player] if type(player) == str else player
    short_players = [p.split(' ')[-1] for p in players]
    player_stats= final_data[(final_data.loc[:,'first_last'].str.contains('|'.join(players)))]
    available_clusters = len(cluster_summary) - player_stats.loc[:,'cluster_rank']
    reduce_clusters = dict(zip(list(set(available_clusters)), scipy.stats.rankdata(list(set(available_clusters)), method='min')))
    y_labels = list(player_stats.loc[:,'cluster_name'])
    cluster_labels, wrapped_labels = [], []
    for label,cluster in zip(y_labels, available_clusters):
        split_word = label.split(' ')
        split_word[-1] = '\n' + split_word[-1]
        if reduce_clusters[cluster] not in cluster_labels:
            wrapped_labels.append(' '.join(split_word))
            cluster_labels.append(reduce_clusters[cluster])

    pstyle.use('fivethirtyeight')
    colours = sns.palettes.color_palette('colorblind',10)
    plt.figure()

    for current_player, colour in zip(players, colours):
        current_player_stats = player_stats[player_stats.loc[:,'first_last'].str.contains(current_player)]
        player_clusters = len(cluster_summary) - current_player_stats.loc[:,'cluster_rank']

        x = current_player_stats.loc[:,'season']
        y = [reduce_clusters[cluster] for cluster in player_clusters]

        temp_change = np.array(y[1:])-np.array(y[:-1]) != 0
        cluster_changes = np.logical_or(np.concatenate((np.array([True]), temp_change)), np.concatenate((temp_change,np.array([True]))))
        x_changes = [x_change for x_change, change in zip(x, cluster_changes) if change]
        y_changes  = [y_change for y_change, change in zip(y, cluster_changes) if change]

        plt.plot(x, y, linewidth=2, alpha=0.7, c=colour)
        plt.scatter(x_changes, y_changes, marker='o', alpha=0.7, c=np.array([colour]))

    ax = plt.gca()
    plt.yticks(ticks=cluster_labels, labels=wrapped_labels, fontsize=8, color='black', fontstyle='normal')
    plt.tick_params(axis='x', labelrotation=45, labelsize=10, labelcolor='gray')
    ax.yaxis.tick_right()
    ax.xaxis.set_major_locator(mtick.MultipleLocator(2))

    if len(players) == 1:
        plt.title(players[0])
    else:
        plt.title('Career Progression')
        plt.legend(labels=short_players, loc=8, ncol=4, fontsize=10, bbox_to_anchor=(0.5,-0.27))
    plt.tight_layout()
    
def show_summary(pickle_data):
    cluster_summary = pickle_data[-1]
    pct_columns = [x for x in cluster_summary.columns if 'pct' in x]
    display_summary = cluster_summary.copy()
    for column in pct_columns:
        display_summary[column] = display_summary[column].map('{:.2f}'.format)
    
    if 'cluster_rank' in display_summary.columns:
        return display_summary.drop('cluster_rank', axis=1)
    else:
        return display_summary

def plot_player(player, final_data, cluster_summary, close_min_max):
    name_column = 'last_name' if len(player.split(' ')) == 1 else 'first_last'
    player_stats= final_data[(final_data.loc[:,name_column] == player)].sort_values('pts', ascending=False).head(1)
    cluster = int(player_stats['cluster'])

    matching_cluster = cluster_summary.loc[cluster,:]
    cluster_index = matching_cluster.name
    full_name = list(player_stats['display_name'])[0]
    close_players = np.resize(np.array(final_data.loc[close_min_max.flatten(), 'display_name']), (len(cluster_summary), 6))
    matching_players = close_players[cluster_index, :]
    stats_only = player_stats[DESIRED_STATS].loc[list(player_stats.index)[0],:]
    player_diff = stats_only - matching_cluster[DESIRED_STATS]
    pct_diff = player_diff / matching_cluster[DESIRED_STATS] * 100

    pstyle.use('default')
    plt.figure()
    labels = [x.replace('_pct','%') for x in pct_diff.index]
    bars = plt.bar(labels, list(pct_diff), edgecolor = 'k', alpha=0.8)
    ax=plt.gca()
    ymin, ymax = plt.ylim()
    yrange = ymax-ymin
    green_pastel = [0.5529, 1.0, 0.5764]
    red_pastel = [1.0, 0.3490, 0.3372]

    for i in range(len(bars)):
        bar = bars[i]
        face_c = red_pastel if bar.get_height() < 0 else green_pastel
        bar.set_color(face_c)
        bar.set_edgecolor('black')

        label = labels[i]
        stat = stats_only[i]
        f = '{:.0%}' if '%' in label else '{:.1f}'
        spacing = min(-yrange * 0.1, -5) 
        ax.text(bar.get_x() + bar.get_width()/2, ymin + spacing*2,
               f.format(stat), ha='center', color='gray', fontsize=10)
        ax.text(bar.get_x() + bar.get_width()/2, ymin + spacing,
               label, ha='center', color='k', fontsize=10)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)

    for spine in ax.spines.values():
        if spine.spine_type == 'bottom':
            spine.set_position('zero')
        elif spine.spine_type != 'left':
            spine.set_visible(False)

    no_bars = len(bars)
    xmin, xmax = plt.xlim()
    xrange = xmax-xmin
    for count, comp_name in zip(range(len(matching_players)), matching_players):
        c = 'gray' if comp_name == full_name else 'black'
        xloc = (0.1 + 0.4 * (count % 3)) * xrange +xmin
        yloc = ymax + yrange * (0.1 - ((count // 3) * 0.05))
        ax.text(xloc, yloc, comp_name, ha='center', fontdict={'color':c})

    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:+.0f}%'))

    if 'cluster_name' in player_stats.columns:
        title_pad = 50
        player_desc = list(player_stats.loc[:,'cluster_name'])[0]
        plt.text(xmin+xrange/2, ymax+yrange*0.18, player_desc, ha='center', fontdict={'color':'dimgrey','size':14})
    else:
        title_pad = 40

    plt.title(full_name, fontdict={'fontsize':20}, pad=title_pad, ha='center')
    plt.tight_layout(pad=1.2)