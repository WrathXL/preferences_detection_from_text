# from embedding import pretrained_embedding
#
# lookup = pretrained_embedding()

# from nltk import word_tokenize
#
# def table_split(table):
#     text = []
#     content = []
#     for i in table:
#         if len(i) == 0:
#             continue
#         if i[0] == '#':
#             text.append(i)
#         else:
#             content.append(i)
#     return text, content
#
# files = ['data/sensitive-3.tsv']
#
# text = []
# for path in files:
#     with open(path) as fd:
#         text += fd.readlines()
#
# t = "".join(text)
# t = t.split('\n\n')
# heading = t[0]+'\n\n\n'
# t = t[1:]
# t[0] = t[0][1:]
# table = [i.split('\n') for i in t]
# sent_index = 0
# word_index = 0
# result = []
#
# for i in table:
#     txt, cnt = table_split(i)
#     for t in txt:
#         sent_index+=1
#         t_len = len(t.split('=')[1].split())
#         c = cnt[word_index: word_index+t_len]
#         for k in range(len(c)):
#             c_t = c[k].split("\t")
#             c[k] = [f'{sent_index}-{k+1}']+c_t[1:]
#             c[k] = '\t'.join(c[k])
#         c = '\n'.join(c)
#         word_index += t_len
#         result.append('\n'.join([t, c]))
#         result.append([t, c])
#     word_index = 0
#
# result = heading + '\n\n'.join(result)
# #
# for path in ['data/sensitive-4.tsv']:
#     with open(path, "w") as fd:
#         fd.write(result)




