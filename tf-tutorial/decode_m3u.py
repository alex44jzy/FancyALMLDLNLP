# -*- coding:utf-8 -*-

import os
import sys
import requests
import datetime
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex


def read_file(file_name):
    with open(file_name) as file:
        data = file.read()
    content_list = data.split('\n')
    return content_list


def get_key_uri(content_list):
    key_line = [item for item in content_list if '#EXT-X-KEY' in item][0]

    def slice_line(line, start, end, symbol='='):
        line = line[start:end]
        return line.split(symbol)[1]

    method_value = slice_line(key_line,
                              key_line.index('METHOD'),
                              key_line.index(','))

    URI_value = slice_line(key_line,
                           key_line.index('URI'),
                           None).strip('\"')
    return method_value, URI_value


def get_ts_path(content_list, key, cipher):
    ts_files_path = []
    count = 0
    for index, line in enumerate(content_list):
        if 'EXTINF' in line:
            ts_files_path.append((count, content_list[index + 1]))
            count += 1
    print("Total ts files: %d" % len(ts_files_path))

    file_no = ts_files_path[0][0]
    res = requests.get(ts_files_path[0][1])

    decode_key = requests.get(key)
    decode_key.encoding = 'utf-8'

    print(decode_key.text)
    # print(len(decode_key.content))


    cryptor = AES.new('54c59472-6ab2-442b-a677-b1b764960e80', AES.MODE_CBC, decode_key.content)
    # with open(os.path.join('ts/', file_no + ".mp4"), 'ab') as f:
    #     f.write(cryptor.decrypt(res.content))


    # print(res.content)



    # if "EXTINF" in line:  # 找ts地址并下载
    #     unknow = False
    #     pd_url = url.rsplit("/", 1)[0] + "/" + file_line[index + 1]  # 拼出ts片段的URL
    #     # print pd_url
    #
    #     res = requests.get(pd_url)
    #     c_fule_name = file_line[index + 1].rsplit("/", 1)[-1]
    #
    #     if len(key):  # AES 解密
    #         cryptor = AES.new(key, AES.MODE_CBC, key)
    #         with open(os.path.join(download_path, c_fule_name + ".mp4"), 'ab') as f:
    #             f.write(cryptor.decrypt(res.content))
    #     else:
    #         with open(os.path.join(download_path, c_fule_name), 'ab') as f:
    #             f.write(res.content)



if __name__ == '__main__':
    fileName = '720.m3u'
    content_list = read_file(fileName)
    method, uri = get_key_uri(content_list)
    get_ts_path(content_list, uri, method)
