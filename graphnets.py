import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from sklearn import model_selection
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GCN, GraphSAGE

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / possible_positives
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / predicted_positives
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def training_split(training):
    train_subjects, test_subjects = model_selection.train_test_split(
        training, train_size=0.2, test_size=None
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=0.5, test_size=None
    )
    print(
        "Array shapes:\n train = {}\n val = {}\n test = {}".format(
            train_subjects.shape, val_subjects.shape, test_subjects.shape
        )
    )
    return train_subjects, val_subjects, test_subjects

def build_model(graphnet, out_layer):
    x_inp, x_out = graphnet.in_out_tensors()
    predictions = layers.Dense(units=out_layer, activation="softmax")(x_out)
    #Model training
    print("Training...")
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.categorical_crossentropy,
        metrics=["acc", f1_m, precision_m, recall_m],
    )
    return model

def plot_results(history):
    plot_hist = history
    plots = ['acc', 'val_acc', 'f1_m', 'val_f1_m', 'loss', 'val_loss']
    plot_hist.history = {key: history.history[key] for key in plots}
    sg.utils.plot_history(plot_hist)

def test_metrics(generator, model, test_subjects):
    #Test Metrics
    test_gen = generator.flow(test_subjects.index, test_subjects.values)
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\n{}: {:0.4f}".format(name, val))

def gcn_pipeline(G, node_subjects, layer_sizes=[16,16], activations=["relu", "relu"]):
    #Train and test split
    train_subjects, val_subjects, test_subjects = training_split(node_subjects)
    
    #GCN training generator
    generator = FullBatchNodeGenerator(G, method="gcn")
    train_gen = generator.flow(train_subjects.index, train_subjects.values,)
    gcn = GCN(
        layer_sizes=layer_sizes, 
        activations=activations, 
        generator=generator, 
        dropout=0.5
    )
    model = build_model(gcn, train_subjects.values.shape[1])
    
    val_gen = generator.flow(val_subjects.index, val_subjects.values)
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    history = model.fit(
        train_gen,
        epochs=200,
        validation_data=val_gen,
        verbose=0,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )
    
    plot_results(history)
    test_metrics(generator, model, test_subjects)
    
        
def graphsage_pipeline(G, node_subjects, layer_sizes=[32,32]):
    train_subjects, val_subjects, test_subjects = training_split(node_subjects)
    
    batch_size = 50
    num_samples = [10, 5]
    generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(train_subjects.index, train_subjects.values, shuffle=True)
    graphsage_model = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.5,
    )

    model = build_model(graphsage_model, train_subjects.values.shape[1])

    val_gen = generator.flow(val_subjects.index, val_subjects.values)
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    history = model.fit(
        train_gen, 
        epochs=200, 
        validation_data=val_gen,
        verbose=0, 
        shuffle=False,
        callbacks=[es_callback]
    )
    
    plot_results(history)
    test_metrics(generator, model, test_subjects)