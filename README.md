<img src="assets/logo.png" alt="EARS logo" title="Environmental Audio Recognition System" align="right" />

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## tOne AI system

**tOne AI** is a proof of concept implementation of a ***convolutional neural network*** for **live environmental audio processing & recognition** on low-power [*SoC*](https://en.wikipedia.org/wiki/System_on_a_chip) devices (at this time it has been developed and tested on a ***Raspberry Pi 3 Model B***).

tOne AIfeatures a background thread for **audio capture & classification** and a [Bokeh](https://github.com/bokeh/bokeh/) server based dashboard providing **live visualization** and **audio streaming** from the device to the browser.

### Caveats: ###

tOne AIis quite taxing on the CPU, so some proper cooling solution (heatsink) is advisable. Nevertheless, when not using the Bokeh app too much, it should work fine even without one.

The live audio stream can get choppy or out-of-sync, especially when using the mute/unmute button.

Actual production deployments would profit from a server-node architecture where SoC devices are only pushing predictions, status updates and audio feeds to a central server handling all end user interaction, material browsing and visualization. This may be implemented in future versions, but no promises here.


## Installation

tOne AIhas been developed and tested on a Raspberry Pi 3 Model B device. To recreate the environment used for developing this demo:

### Step 1 - prepare a Raspberry Pi device
- Get a spare Raspberry Pi 3 Model B with a blank SD card.
- Install a Raspbian Jessie Lite distribution (tested on version April 2017):
  - Download a Raspbian Jessie Lite image from [RaspberryPi.org](https://www.raspberrypi.org/downloads/raspbian/).
  - Use Etcher to flash the SD card (see [Raspberry Pi docs](https://www.raspberrypi.org/documentation/installation/installing-images/README.md) for details).
- Boot the device with the new card.
- Attach some input & display devices for configuration.
- Login using default credentials (`user: pi, password: raspberry`).
- Setup Wi-Fi access (see [Wi-Fi config on Raspberry Pi](https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md)).
- Use `sudo raspi-config` to enable SSH.
- Recreate SSH host keys:

```bash
sudo rm /etc/ssh/ssh_host_*
sudo dpkg-reconfigure openssh-server
```
  
### Step 2 - install Python 3.6 using [Berry Conda](https://github.com/jjhelmus/berryconda)

- Install conda for armv7l to `/opt/conda`:

```bash
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
sudo md5sum Miniconda3-latest-Linux-armv7l.sh
sudo /bin/bash Miniconda3-latest-Linux-armv7l.sh
```
 
- Add `export PATH="/opt/conda/bin:$PATH"` to the end of `/home/pi/.bashrc`. Then reload with `source /home/pi/.bashrc`.
```bash
echo 'PATH=/opt/conda/bin:$PATH; export PATH' >> ~/.bashrc
source ~/.bashrc
```
- Install Python with required packages:

```bash
conda config --add channels rpi
conda create -n tOne AIpython=3.6
source activate tOne AI
conda install cython numpy pandas scikit-learn cffi h5py
```

- Make sure PortAudio headers are available. If not, installing pyaudio will complain later on:

```bash
sudo apt-get install portaudio19-dev
```

### Step 3 - download tOne AIand install requirements

- Download tOne AIsource code and unpack it to `/home/pi/tOne AI`. Then install the required packages by issuing:

```bash
pip install -r /home/pi/tOne AI/requirements.txt
```

- Plug a microphone into the USB port (or some other audio device, but that's the one I used for initial testing), switch it into an audio interface mode (44.1 kHz/16 bit), and verify it's listed by `python -m sounddevice`.
- Update the `--allow-websocket-origin` option inside `/home/pi/tOne AI/run.sh` file with the IP address of the Raspberry Pi device.
- Finally, run the app with:

```bash
chmod +x /home/pi/tOne AI/run.sh
cd /home/pi/tOne AI
./run.sh
```

- Point the web browser to: `http://RASPBERRY_PI_IP:5006/`

## Training new models


If you want to train the same model on a different dataset:
- Download the source code to a workstation/server with a GPU card.
- Put all audio files (WAV) into `tOne AI/dataset/audio`.
- Replace the [`tOne AI/dataset/dataset.csv`](tOne AI/dataset/dataset.csv) file with new CSV:

```csv
filename,category
```

- Run `python train.py` - this should result in the following files being generated on the server:

File                | Description
------------------- | ------------------------------------------------------- 
`model.h5`          | weights of the learned model
`model.json`        | a serialized architecture of the model (Keras >=2.0.0)  
`model_labels.json` | dataset labels

- Upload the new model files to the Raspberry Pi device and restart the app.

If you want to train a completely different model, then you can have a look at [`train.py`](tOne AI/train.py). In this case you probably know what to do either way.


## License

MIT © Mykhailo Kunynets
