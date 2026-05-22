import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import plotly.express as px


from src.model import ImageClassifier
from src.trainer import ModelTrainer
from src.dataset import DatasetImmobilierSoCal

st.title("Image Classifier")

if "history" not in st.session_state:
    st.session_state.history = [] 
if "y_true" not in st.session_state:
    st.session_state.y_true = []
if "y_pred" not in st.session_state:
    st.session_state.y_pred = []

tab1, tab2, tab3, tab4 = st.tabs([
    "Classifier testing", 
    "Training Performances", 
    "Classification report", 
    "Benchmark & Comparative Analysis"
])

#
with tab1:
    st.subheader("Load a picture of house to analyze it")
    
    file_upload = st.file_uploader(
        label="Input a house picture...",
        type=["png", "jpg", "webp", "jpeg"]
    )

    mes_transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    @st.cache_resource
    def load_model():
        model = ImageClassifier()

        try:
            model.load_state_dict(torch.load("ImageClassifier_v1.pth", map_location="cpu"))
        except FileNotFoundError:
            st.warning("Aucun modèle pré-entraîné trouvé. L'IA n'est pas initialisée.")
        model.eval()
        return model

    ai_model = load_model()

    real_estate_classes = ["Luxury house", "Standard house", "Renovation house"]

    if file_upload is not None:
        pil_image = Image.open(file_upload)
        st.image(pil_image, caption="Imported image", use_container_width=True)

        if st.button("Analyze this house..."):
            rgb_image = pil_image.convert("RGB")
            image_tensor = mes_transformations(rgb_image).unsqueeze(0)

            with torch.no_grad():
                predictions = ai_model(image_tensor)
            
            index_prediction = torch.argmax(predictions, dim=1).item()
            result = real_estate_classes[index_prediction % 3]
            
            st.success(f"Analysis Result : **{result}**")

            data_saved = {
                "file": file_upload,
                "name": file_upload.name,
                "prediction": result
            }

            noms_existants = [item["name"] for item in st.session_state.history]
            if data_saved["name"] not in noms_existants:
                st.session_state.history.append(data_saved)
    
    st.subheader("History :")
    for item in st.session_state.history:
        col_icon, col_text = st.columns([1, 4])
        with col_icon:
            st.image(item["file"], width=60) 
        with col_text:
            st.write(f"**{item['name']}**")
            st.caption(f"Predicted as: {item['prediction']}")



@st.cache_resource
def prepare_dataloaders():

    df_full = pd.read_csv('data/socal2.csv')

    def calculate_label(prix):
        if prix >= 1000000: return 0
        elif prix <= 400000: return 1
        else: return 2

    df_full['label'] = df_full['price'].apply(calculate_label)

    transf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    df_train, df_test = train_test_split(
        df_full,
        test_size=0.2,
        random_state=42,
        stratify=df_full['label']
    )

    dataset_train = DatasetImmobilierSoCal(dataframe_deja_pret=df_train,
                                        img_dir='data/socal2/socal_pics',
                                        transform=transf)
    dataset_test = DatasetImmobilierSoCal(dataframe_deja_pret=df_test,
                                        img_dir='data/socal2/socal_pics',
                                        transform=transf)

    dl_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dl_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    return dl_train, dl_test



with tab2: 
    st.header("Training Performances")


    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = {"Train Loss": [], "Test Accuracy": []}


    try:
        dataloader_train, dataloader_test = prepare_dataloaders()
        st.success(f"Data loaded : {len(dataloader_train.dataset)} training images, {len(dataloader_test.dataset)} test images.")
    except Exception as e:
        st.error(f"Error during data loading. Verify the path of data/socal2.csv : {e}")
        st.stop() 


    model_train = ImageClassifier()
    optimizer = torch.optim.Adam(model_train.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainer = ModelTrainer(model_train, optimizer, criterion, device='cpu')


    epochs = st.slider("Number of epochs for training :", min_value=1, max_value=20, value=5)

    if st.button("Load the training"):
        

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col_metric1, col_metric2 = st.columns(2)
        loss_metric = col_metric1.empty()
        acc_metric = col_metric2.empty()
        

        col_graph1, col_graph2 = st.columns(2)
        graph_loss_placeholder = col_graph1.empty() 
        graph_acc_placeholder = col_graph2.empty() 


        for epoch in range(epochs):
            status_text.info(f"Epoch {epoch+1}/{epochs} running...")
            
            train_metrics = trainer.train_epoch(dataloader_train)
            current_loss = train_metrics["loss"]
            
            test_metrics = trainer.test(dataloader_test)
            current_test_acc = test_metrics["accuracy"] 


            st.session_state.metrics_history["Train Loss"].append(current_loss)
            st.session_state.metrics_history["Test Accuracy"].append(current_test_acc)

            if epoch == epochs - 1 :
                st.session_state.y_true = test_metrics["y_true"]
                st.session_state.y_pred = test_metrics["y_pred"]


            progress_bar.progress((epoch + 1) / epochs)
            loss_metric.metric(label="Train Loss", value=f"{current_loss:.4f}")
            acc_metric.metric(label="Test Accuracy", value=f"{current_test_acc:.2f}%")

            df = pd.DataFrame(st.session_state.metrics_history)
            
            fig_loss = px.line(df, y="Train Loss", title="Évolution de la Perte (Loss)", markers=True)
            fig_loss.update_traces(line_color="red")
            
            fig_acc = px.line(df, y="Test Accuracy", title="Évolution de la Précision (%)", markers=True)
            fig_acc.update_traces(line_color="green")

            graph_loss_placeholder.plotly_chart(fig_loss, use_container_width=True)
            graph_acc_placeholder.plotly_chart(fig_acc, use_container_width=True)

        status_text.success("Training finished with success !")
        

        trainer.save_model("ImageClassifier_v1.pth")
        st.info("Model saved named ImageClassifier_v1.pth")


    elif len(st.session_state.metrics_history["Train Loss"]) > 0:
        df = pd.DataFrame(st.session_state.metrics_history)
        
        col_graph1, col_graph2 = st.columns(2)
        
        fig_loss = px.line(df, y="Train Loss", title="Loss Evolution", markers=True)
        fig_loss.update_traces(line_color="red")
        col_graph1.plotly_chart(fig_loss, use_container_width=True)
        
        fig_acc = px.line(df, y="Test Accuracy", title="Accuracy  (%)", markers=True)
        fig_acc.update_traces(line_color="green")
        col_graph2.plotly_chart(fig_acc, use_container_width=True)


with tab3: 
    st.header("Classification Metrics")


    if len(st.session_state.y_pred) > 0:
        st.subheader("Final Performance Report")

        from sklearn.metrics import classification_report, confusion_matrix

        report_dict = classification_report(
            st.session_state.y_true, 
            st.session_state.y_pred, 
            target_names=real_estate_classes,
            output_dict=True
        )
        
        df_report = pd.DataFrame(report_dict).transpose()
        

        st.dataframe(df_report.style.format(precision=2), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive Confusion Matrix")

        matrix = confusion_matrix(st.session_state.y_true, st.session_state.y_pred)

        fig_matrix = px.imshow(
            matrix,
            text_auto=True, 
            labels=dict(x="Predicted Classes", y="Real Classes", color="Houses numer"),
            x=real_estate_classes,
            y=real_estate_classes,
            color_continuous_scale="Blues",
            title="Confusion Matrix (Real values vs Predictions)"
        )
        

        st.plotly_chart(fig_matrix, use_container_width=True)

    else:

        st.info("None Result Available. Please start a full training session in advance in the 'Training Performances' tab..")


    








with tab4: 
    st.header("Model Benchmarking")