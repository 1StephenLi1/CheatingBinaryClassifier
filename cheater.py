import helper
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
def transform_data(data,v):
    with open(data,'r') as file:
        data_list=[line.strip().split(' ') for line in file]
    paragraphs=[]
    for para in data_list:
        little_paragraph = ""
        for word in para:
            little_paragraph += word
            little_paragraph += " "
        paragraphs.append(little_paragraph)
    x_vector = TfidfVectorizer(vocabulary=v,token_pattern='[^\s]+')
    x_data = x_vector.fit_transform(paragraphs)
    return x_data,data_list
def rm_all(list,val):
    while val in list:
        list.remove(val)
def fool_classifier(test_data): ## Please do not change the function defination...
    strategy_instance=helper.strategy()
    parameters={}
    list_class0 = strategy_instance.class0
    list_class1 = strategy_instance.class1
    vertical_dim_of_trainx = len(list_class0) + len(list_class1)
    paragraphs = []
    for para in list_class0:
        little_paragraph = ""
        for word in para:
            little_paragraph += word
            little_paragraph += " "
        paragraphs.append(little_paragraph)
    for para in list_class1:
        little_paragraph = ""
        for word in para:
            little_paragraph += word
            little_paragraph += " "
        paragraphs.append(little_paragraph)
    y_train = []
    for i in range(len(list_class0)):
        y_train.append(0)
    for j in range(len(list_class1)):
        y_train.append(1)
    x_vector = TfidfVectorizer(token_pattern='[^\s]+')
    x_train = x_vector.fit_transform(paragraphs)
    words_bag=x_vector.vocabulary_
    # Looking for the best 'C'
    C_parameter =np.arange(0.01, 1.2, 0.01)
    parameters_for_grid = {'kernel':['linear'], 'C':C_parameter}
    clf_for_grid = GridSearchCV(svm.SVC(), parameters_for_grid)
    clf_for_grid.fit(x_train, y_train)
    c_best=clf_for_grid.best_params_
    word_list= x_vector.get_feature_names()
    parameters={'kernel':'linear', 'C':c_best['C'], 'degree':1, 'coef0':0, 'gamma': 'auto'}
    clf = strategy_instance.train_svm(parameters, x_train, y_train)
    weight_list=clf.coef_.toarray().tolist()[0]
    for i in range(len(weight_list)):
        if weight_list[i]>0:
            weight_list[i]=weight_list[i]*2
    x_data,data_list=transform_data(test_data,words_bag)
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    dict_for_word = {}
    for idx in range(len(word_list)):
        dict_for_word[word_list[idx]] = weight_list[idx]
    sorted_dict_for_word = sorted(dict_for_word.items(), key=lambda x:x[1])
    reversed_dict_for_word = sorted(dict_for_word.items(), key=lambda x:x[1], reverse=True)
    list_for_test_dict = []
    sorted_test_dict=[]
    rsorted_test_dict=[]
    for idx in range(len(data_list)):
        paragraph_list = data_list[idx]
        test_data_dict = {}
        for word in paragraph_list:
            if word in dict_for_word:
                test_data_dict[word] = dict_for_word[word]
        list_for_test_dict.append(test_data_dict)
        sorted_test_dict.append(sorted(test_data_dict.items(), key=lambda x:x[1]))
        rsorted_test_dict.append(sorted(test_data_dict.items(), key=lambda x:x[1],reverse=True))

    for i in range(len(list_for_test_dict)):
        time=20
        s_j=0
        add_index=0
        while(time>0):
            if sorted_dict_for_word[add_index][1]+rsorted_test_dict[i][s_j][1]>0 :
                rm_all(data_list[i],(rsorted_test_dict[i][s_j][0]))
                time-=1
                s_j+=1
            else:
                if sorted_dict_for_word[add_index][0] not in data_list[i]:
                    data_list[i].append(sorted_dict_for_word[add_index][0])
                    time-=1
                add_index+=1
    f=open('modified_data.txt','w')
    for i in data_list:
        line=''
        for word in i:
            line+=word
            line+=' '
        line+='\n'
        f.write(line)
    f.close()
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    modify_x,m_list=transform_data('modified_data.txt',words_bag)
    return strategy_instance
fool_classifier('./test_data.txt')










