#!/usr/bin/python3

import pandas as pd

from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

def es_conn(host):
    es = Elasticsearch(
    ["192.168.0.56"],
    scheme="http",
    port=9200,
    http_auth=('admin', 'admin'),
    verify_certs=False,
    ssl_show_warn=False)

    # print(es.info())

    return es

es = es_conn("192.168.0.56")

# twmiot_es_keys = list(es.indices.get("twmiot*").keys())
# actives_es_keys = list(es.indices.get("actives*").keys())
# alarms_keys = list(es.indices.get("alarms*").keys())

# get last index
# print(twmiot_es_keys[-1])
# print(actives_es_keys[-1])

s = Search(using=es, index="twmiot-*")

# s = Search(using=es, index="twmiot-*") \
#     .filter("range", **{'@timestamp' : {"gte": "now-2d/d","lt": "now/d"} }) \
#     .query("term", Sites_ID="63") \
#     .sort({"@timestamp": {"order": "desc"}})

##### aggrs
s.aggs.bucket('group', 'terms', field="grpId", size=0) \
.bucket('twitter_count', 'range', field='twitter.followers',
    range={'from': 'now-2d/d', 'to': 'now/d'}
) \
.bucket('input_oct', 'sum', field='delta_input', size=0) \
.bucket('output_oct', 'sum', field='delta_output', size=0)

print(s.to_dict())

##### query
# response = s.params(size=1440).execute()
# print(response["hits"]["total"]["value"])

##### scan
# response =[]

# for hit in s.params(size=1,scroll="2m").scan():
#     print(hit.to_dict())
#     response.append(hit)

# print(len(response))

# for post in results:
#     print(post)

# exit(0)
# df = pd.DataFrame(results)

# print(df)