path="data2/datasets/ruoguli/idl_project_datas/baseline_model"
mkdir -p $path
cd $path

# Download pre-trained model
for model in "PS-FCN_B_S_32.pth.tar" "UPS-FCN_B_S_32.pth.tar"; do
    wget http://www.visionlab.cs.hku.hk/data/PS-FCN/models/${model}
done

