from waveNetModel import WavenetUnconditional
from audioDataset import AudioDataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import time

if __name__ == "__main__":
    model_save_path = "out/model_save/" + time.strftime("%Y%m%d%H%M%S") + "uncondWavenet.pt"
    dataset = AudioDataset("piano/",test=True, layer_num=5, target_field=128)
    print(dataset[0][0].shape, dataset[0][1].shape)
    train_dataset, val_dataset = random_split(dataset, [0.9,0.1])

    train_dataLoader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = WavenetUnconditional(audio_channel_num = 1, target_field=128, num_layers=5)
    print(model)
    # trainer = pl.Trainer(max_epochs=2)
    # trainer.fit(model, 
    #             train_dataloaders=train_dataLoader,
    #             val_dataloaders=val_dataloader
    #             )
    
    # torch.save(model.state_dict(), model_save_path)
    
    # for i in range(5):
    #     # generate 5 sample audios
    #     model.generate_audio(file_name = "uncondWavenet" + str(i) + ".wav")
    

