FROM ultralytics/ultralytics:8.3.202

WORKDIR /app

# COPY req_yolo.txt .
# RUN pip install -r req_yolo.txt --no-cache-dir
RUN pip install geopandas==1.0.1, jupyter==1.1.1, jupyterlab==4.4.7, loguru==0.7.3, plotly==6.2.0, scikit-image==0.25.2, scikit-learn==1.7.1 --no-cache-dir
RUN pip install ray[tune]==2.47.1, tensorboard==2.19.0 --no-cache-dir

COPY scripts/ scripts/