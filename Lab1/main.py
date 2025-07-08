import numpy as np
import matplotlib.pyplot as plt
import argparse, sys

class SimpleTwoLayerModel:
    def __init__(self, l1=9, l2=4, learning_rate=0.1, activitation_type="sigmoid", optimizer_type   ="GD", epoch=10000):
        super(SimpleTwoLayerModel, self).__init__()
        self.w1 = np.random.rand(2, l1)
        self.w2 = np.random.rand(l1, l2)
        self.w3 = np.random.rand(l2, 1)
        self.learning_rate = learning_rate
        self.loss_list = []
        self.activitation_type = activitation_type
        self.optimizer_type = optimizer_type
        self.epoch = epoch
        self.y_pred_binary = None
        self.beta = 0.7
        self.v1 = None
        self.v2 = None
        self.v3 = None
        if self.optimizer_type == "momentum":
            self.v1 = np.zeros_like(self.w1)
            self.v2 = np.zeros_like(self.w2)
            self.v3 = np.zeros_like(self.w3)
        
    def train(self, x, y):
        for i in range(self.epoch):
            # forword pass
            # layer 1
            a1 = x@self.w1
            z1 = activitate(self.activitation_type, a1)

            # layer 2
            a2 = z1@self.w2
            z2 = activitate(self.activitation_type, a2)

            # output
            a3 = z2@self.w3
            y_train_pred = sigmoid(a3)
            loss = calculate_loss(y_train_pred, y)
            self.loss_list.append(np.mean(loss))
            if(i%500==0):
                print(f"epoch: {i}, loss: {np.mean(loss)}")
            # backword propagation
            #output to layer2
            d_L_y = (y_train_pred-y)
            d_y_a3 = sigmoid(a3)*(1-sigmoid(a3))
            d_L_a3 = d_L_y*d_y_a3
            w3_loss = z2.T@d_L_a3

            #layer2 to layer1
            d_L_z2 = d_L_a3@self.w3.T
            d_z2_a2 = d_activitate(self.activitation_type, a2)
            d_L_a2 = d_L_z2*d_z2_a2
            w2_loss = z1.T@d_L_a2

            #layer1 to intput
            d_L_z1 = d_L_a2@self.w2.T
            d_z1_a1 = d_activitate(self.activitation_type, a1)
            d_L_a1 = d_L_z1*d_z1_a1
            w1_loss = x.T@d_L_a1

            # update weight
            self.w1, self.v1 = optimizer(self.optimizer_type, self.w1, self.v1, self.learning_rate, self.beta, w1_loss)
            self.w2, self.v2 = optimizer(self.optimizer_type, self.w2, self.v2, self.learning_rate, self.beta, w2_loss)
            self.w3, self.v3 = optimizer(self.optimizer_type, self.w3, self.v3, self.learning_rate, self.beta, w3_loss)
    
    def test(self, x, y):
        a1 = x@self.w1
        z1 = activitate(self.activitation_type, a1)

        # layer 2
        a2 = z1@self.w2
        z2 = activitate(self.activitation_type, a2)

        # output
        a3 = z2@self.w3
        y_final_pred = sigmoid(a3)
        final_loss = calculate_loss(y, y_final_pred)
        self.y_pred_binary = (y_final_pred>0.5).astype(int)
        correct_predictions = np.sum(self.y_pred_binary == y)

        accuracy = correct_predictions / len(y)
        for i in range(len(x)):
            # Spec 格式是 Iter91, Iter92... 但這裡我們從 0 開始
            print(f"Iter{i: <4} | Ground truth: {y[i][0]:.1f} | prediction: {y_final_pred[i][0]:.5f}")

        print("-" * 50)
        print(f"loss={final_loss:.5f} accuracy={accuracy*100:.2f}%")
        print("-" * 50)
    
    def loss_curve(self, name):
        plt.plot(self.loss_list, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} Training Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{name}_loss_plot.png")  # 儲存圖片
        plt.show()
                
def activitate(activitate_type, input_data):
    if activitate_type == "sigmoid":
        return sigmoid(input_data)
    if activitate_type == "relu":
        return relu(input_data)
    if activitate_type == "tanh":
        return tanh(input_data)
    
def d_activitate(activitate_type, input_data):
    if activitate_type == "sigmoid":
        return sigmoid(input_data)*(1-sigmoid(input_data))
    if activitate_type == "relu":
        return (input_data>0).astype(float)
    if activitate_type == "tanh":
        return 1-np.square(tanh(input_data))
    
def optimizer(optimizer_type, w, v, learning_rate, beta, w_loss):
    if optimizer_type == "GD":
        return w-learning_rate*w_loss, v
    elif optimizer_type == "momentum":
        v_new = beta*v+(1-beta)*w_loss
        w_new = w-learning_rate*v_new
        return w_new, v_new

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def calculate_loss(y, y_hat):
    return np.mean((y-y_hat)**2)

def plot_comparison_scatter(X, y_true, y_pred_binary, main_title, filename):
    plt.figure(figsize=(12, 5.5))
    # --- 左圖: Ground Truth ---
    plt.subplot(1, 2, 1)
    # 找出 Class 0 和 Class 1 的索引
    class0_idx_true = (y_true.ravel() == 0)
    class1_idx_true = (y_true.ravel() == 1)
    
    plt.scatter(X[class0_idx_true, 0], X[class0_idx_true, 1], c='blue', label='Class 0')
    plt.scatter(X[class1_idx_true, 0], X[class1_idx_true, 1], c='red', label='Class 1')
    
    plt.title("Ground Truth", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- 右圖: Prediction Result ---
    plt.subplot(1, 2, 2)
    # 找出預測為 Class 0 和 Class 1 的索引
    class0_idx_pred = (y_pred_binary.ravel() == 0)
    class1_idx_pred = (y_pred_binary.ravel() == 1)

    plt.scatter(X[class0_idx_pred, 0], X[class0_idx_pred, 1], c='blue', label='Predicted 0')
    plt.scatter(X[class1_idx_pred, 0], X[class1_idx_pred, 1], c='red', label='Predicted 1')
    
    plt.title("Prediction Result", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 總標題和儲存
    plt.suptitle(main_title, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 調整佈局以防標題重疊
    plt.savefig(filename)
    plt.show()

def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("-l1", type=int, default=9, help="number of units in layer 1")
    parser.add_argument("-l2", type=int, default=4, help="number of units in layer 2")
    parser.add_argument("-act", type=str, default="sigmoid", help="activation type(sigmoid, relu, tanh)")
    parser.add_argument("-opt", type=str, default="GD", help="optimizer type(GD, momentum)")
    parser.add_argument("-epoch", type=int, default=10000, help="number of epochs")
    parser.add_argument("-data", type=str, default="linear", help="data type(linear, xor)")
    args = parser.parse_args()
    if args.data == "linear":
        x, y = generate_linear(n=100)
        x_test, y_test = generate_linear(n=100)
    elif args.data == "xor":
        x, y = generate_XOR_easy()
        x_test, y_test = generate_XOR_easy()
    else:
        error_message = f"錯誤：無效的資料集類型 '{args.data}'。請選擇 'linear' 或 'xor'。"
        sys.exit(error_message)
    simpleNN = SimpleTwoLayerModel(l1=args.l1, l2=args.l2, learning_rate=args.lr, activitation_type=args.act, optimizer_type=args.opt, epoch=args.epoch)
    simpleNN.train(x, y)
    simpleNN.test(x_test, y_test)
    simpleNN.loss_curve(args.data)
    plot_comparison_scatter(x_test, y_test, simpleNN.y_pred_binary, f"{args.data} Model Results", f"{args.data}_comparison_scatter.png")