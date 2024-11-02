import matplotlib.pyplot as plt

def read_vector_file(file_path):
    x_values = []
    y_values = []
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            y = float(line)  
            x_values.append(idx)  
            y_values.append(y)
    return x_values, y_values

def plot_vectors(vector_files):
    list1 = ['10','20','40','80×90','160×180']
    i=0
    for file_path in vector_files:
        x, y = read_vector_file(file_path)
        plt.plot(x, y, label=list1[i])
        i+=1
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Vector Plot')
    plt.legend()
    plt.show()


vector_files = ['Data10.txt', 'Data20.txt', 'Data40.txt', 'Data80×90.txt','DataClosure160×180.txt']
plot_vectors(vector_files)