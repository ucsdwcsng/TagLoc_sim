import sys
var_names=['train_filenames','test_filenames','valid_filenames','depth','depth_step','eta','n_epochs','result_dir','sigma','model','channels']
def get_config_from_file(fn):
    with open(fn,'r') as f:
        data=f.readlines()
    cur_var=''
    config=dict()
    for line in data:
        #print(line)
        if(line[0]=='#'):
            continue
        else:
            #print(len(var_names))	
            line=line[:-1]
            idx=[i for i in range(len(var_names)) if var_names[i]==line]
            if(len(idx)==1):
                #print(idx)
                #print(var_names[idx[0]])
                cur_var=var_names[idx[0]]
                config[cur_var]=[]
            else:
                config[cur_var].append(line)
    print(config)
    return config                
        

#c=get_config_from_file('config8.txt')
#print(c)
