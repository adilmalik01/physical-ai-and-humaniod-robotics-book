---
sidebar_position: 2
---

# Chapter 02: Speech Recognition with OpenAI Whisper

## Implementing Robust Speech-to-Text for Humanoid Robots

In this chapter, we'll explore how to implement speech recognition in humanoid robots using OpenAI Whisper, the state-of-the-art automatic speech recognition (ASR) system. Whisper provides exceptional accuracy and robustness, making it ideal for human-robot interaction in noisy environments.

### Understanding OpenAI Whisper

OpenAI Whisper is a general-purpose speech recognition model that demonstrates strong performance across various domains, including robotics applications. Key features include:

#### Model Architecture
- **Transformer-based**: Uses a transformer encoder-decoder architecture
- **Multilingual**: Supports 99+ languages out of the box
- **Robust**: Performs well on various accents, background noise, and technical speech
- **Open-source**: Available under MIT license for research and commercial use

#### Performance Characteristics
- **High Accuracy**: State-of-the-art performance on various benchmarks
- **Noise Robustness**: Maintains accuracy in noisy environments
- **Real-time Capability**: Can process audio in real-time with proper optimization
- **Zero-shot Learning**: Works well on domains it wasn't explicitly trained on

### Whisper Installation and Setup

To use Whisper in your humanoid robot system:

```bash
# Install Whisper and its dependencies
pip install openai-whisper

# Install additional dependencies for audio processing
pip install torch torchaudio
pip install pyaudio sounddevice  # For audio capture
```

### Basic Whisper Implementation

Here's a basic implementation of Whisper for speech recognition:

```python
import whisper
import torch
import numpy as np
import pyaudio
import wave
import threading
import queue
import time

class WhisperSpeechRecognizer:
    """
    Speech recognition using OpenAI Whisper
    """
    def __init__(self, model_size="base", device="cuda"):
        """
        Initialize Whisper speech recognizer

        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cuda', 'cpu')
        """
        self.model_size = model_size
        self.device = device

        # Load Whisper model
        self.model = whisper.load_model(model_size, device=device)

        # Audio parameters
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1

        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Processing queue
        self.audio_queue = queue.Queue()

        # Wake word detection (optional)
        self.wake_word = "hey robot"
        self.wake_word_detected = False

        # Callbacks
        self.speech_callbacks = []

        print(f"Whisper model '{model_size}' loaded on {device}")

    def start_listening(self):
        """Start audio capture and processing"""
        # Open audio stream
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Start audio processing thread
        self.listening_thread = threading.Thread(target=self._audio_processing_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()

        print("Started listening...")

    def stop_listening(self):
        """Stop audio capture and processing"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _audio_processing_loop(self):
        """Main audio processing loop"""
        audio_buffer = []
        buffer_duration = 3.0  # Process 3 seconds of audio at a time

        while True:
            try:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Add to buffer
                audio_buffer.extend(audio_chunk)

                # Process buffer when it reaches desired duration
                if len(audio_buffer) >= int(self.sample_rate * buffer_duration):
                    # Process the audio buffer
                    self._process_audio_buffer(audio_buffer[:int(self.sample_rate * buffer_duration)])

                    # Keep overlap for continuity
                    overlap = int(self.sample_rate * 0.5)  # 0.5 second overlap
                    audio_buffer = audio_buffer[-overlap:]

            except Exception as e:
                print(f"Error in audio processing: {e}")
                time.sleep(0.1)

    def _process_audio_buffer(self, audio_buffer):
        """Process audio buffer with Whisper"""
        try:
            # Convert to tensor and ensure proper format
            audio_tensor = torch.from_numpy(np.array(audio_buffer)).to(self.device)

            # Transcribe audio using Whisper
            result = self.model.transcribe(
                audio_tensor,
                language='en',
                task='transcribe',
                fp16=torch.cuda.is_available()
            )

            # Check if transcription is meaningful
            if result['text'].strip() and len(result['text'].strip()) > 3:
                # Call registered callbacks
                for callback in self.speech_callbacks:
                    callback(result['text'].strip(), result)

        except Exception as e:
            print(f"Error transcribing audio: {e}")

    def add_speech_callback(self, callback):
        """Add callback for speech recognition results"""
        self.speech_callbacks.append(callback)

    def transcribe_audio_file(self, audio_file_path):
        """Transcribe an audio file"""
        result = self.model.transcribe(audio_file_path)
        return result['text']

    def transcribe_audio_buffer(self, audio_buffer):
        """Transcribe an audio buffer directly"""
        audio_tensor = torch.from_numpy(audio_buffer).to(self.device)
        result = self.model.transcribe(audio_tensor)
        return result['text']
```

### Advanced Whisper Implementation for Robotics

For humanoid robot applications, we need enhanced functionality:

```python
import whisper
import torch
import numpy as np
import pyaudio
import threading
import queue
import time
from dataclasses import dataclass
from typing import Callable, List, Optional
import webrtcvad  # Voice Activity Detection

@dataclass
class SpeechResult:
    """Data class for speech recognition results"""
    text: str
    confidence: float
    timestamp: float
    language: str
    segments: List[dict]

class RobustWhisperRecognizer:
    """
    Enhanced Whisper implementation for robotics applications
    """
    def __init__(self, model_size="base", device="cuda",
                 sensitivity=2, vad_aggressiveness=3):
        """
        Initialize robust Whisper recognizer

        Args:
            model_size: Whisper model size
            device: Computation device
            sensitivity: VAD sensitivity (0-3, higher = more sensitive)
            vad_aggressiveness: VAD aggressiveness (0-3, higher = less aggressive)
        """
        self.model = whisper.load_model(model_size, device=device)
        self.device = device

        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 320  # 20ms chunks for VAD
        self.audio_format = pyaudio.paInt16
        self.channels = 1

        # Voice Activity Detection
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Speech detection parameters
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds
        self.max_silence_duration = 1.0  # Maximum silence before stopping

        # Processing queues
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # State management
        self.is_listening = False
        self.speech_detected = False
        self.current_audio_buffer = []

        # Callbacks
        self.speech_callbacks = []
        self.vad_callbacks = []

        # Performance monitoring
        self.transcription_times = []

        print(f"Robust Whisper recognizer initialized on {device}")

    def start_listening(self):
        """Start audio capture with VAD"""
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.is_listening = True

        # Start audio processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("Started listening with VAD...")

    def stop_listening(self):
        """Stop audio capture"""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _process_audio_stream(self):
        """Process audio stream with VAD"""
        speech_buffer = []
        silence_count = 0
        speech_count = 0

        while self.is_listening:
            try:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                # Convert to numpy array
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Perform VAD
                is_speech = self.vad.is_speech(data, self.sample_rate)

                if is_speech:
                    # Add to speech buffer
                    speech_buffer.extend(audio_chunk)
                    speech_count += 1
                    silence_count = 0  # Reset silence counter

                    # Notify VAD callbacks
                    for callback in self.vad_callbacks:
                        callback('speech_detected', len(speech_buffer))

                else:
                    # Check if we have accumulated speech to process
                    if len(speech_buffer) > 0:
                        silence_count += 1

                        # If enough silence has passed, process the speech
                        if (silence_count * self.chunk_size / self.sample_rate) >= self.max_silence_duration:
                            if self._is_valid_speech(speech_buffer):
                                # Process speech buffer
                                self._process_speech_buffer(speech_buffer)

                            # Clear buffer
                            speech_buffer = []
                            speech_count = 0

                    # Notify VAD callbacks
                    for callback in self.vad_callbacks:
                        callback('silence_detected', silence_count)

            except Exception as e:
                print(f"Error in audio stream processing: {e}")
                time.sleep(0.1)

    def _is_valid_speech(self, audio_buffer):
        """Check if audio buffer contains valid speech"""
        duration = len(audio_buffer) / self.sample_rate
        return duration >= self.min_speech_duration

    def _process_speech_buffer(self, audio_buffer):
        """Process speech buffer with Whisper"""
        start_time = time.time()

        try:
            # Convert to float32 and normalize
            audio_float = np.array(audio_buffer, dtype=np.float32) / 32768.0

            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio_float).to(self.device)

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_tensor,
                language='en',
                task='transcribe',
                fp16=torch.cuda.is_available()
            )

            # Calculate processing time
            processing_time = time.time() - start_time
            self.transcription_times.append(processing_time)

            # Filter out empty or low-confidence results
            if result['text'].strip() and len(result['text'].strip()) > 3:
                # Create speech result object
                speech_result = SpeechResult(
                    text=result['text'].strip(),
                    confidence=self._estimate_confidence(result),
                    timestamp=time.time(),
                    language=result.get('language', 'en'),
                    segments=result.get('segments', [])
                )

                # Call registered callbacks
                for callback in self.speech_callbacks:
                    callback(speech_result)

                print(f"Transcribed: '{speech_result.text}' (Confidence: {speech_result.confidence:.2f}, Time: {processing_time:.2f}s)")

        except Exception as e:
            print(f"Error processing speech buffer: {e}")

    def _estimate_confidence(self, result):
        """Estimate confidence of transcription"""
        # Simple confidence estimation based on result properties
        # In practice, this could use more sophisticated methods
        if 'text' in result and result['text']:
            # Rough confidence based on text length and other factors
            return min(1.0, len(result['text']) / 100.0 + 0.5)
        return 0.0

    def add_speech_callback(self, callback: Callable[[SpeechResult], None]):
        """Add callback for speech recognition results"""
        self.speech_callbacks.append(callback)

    def add_vad_callback(self, callback: Callable[[str, int], None]):
        """Add callback for voice activity detection events"""
        self.vad_callbacks.append(callback)

    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.transcription_times:
            return {}

        avg_time = sum(self.transcription_times) / len(self.transcription_times)
        max_time = max(self.transcription_times)
        min_time = min(self.transcription_times)

        return {
            'avg_transcription_time': avg_time,
            'max_transcription_time': max_time,
            'min_transcription_time': min_time,
            'total_transcriptions': len(self.transcription_times),
            'current_real_time_factor': avg_time * self.sample_rate / len(self.transcription_times) if self.transcription_times else 0
        }
```

### Integration with ROS 2

To integrate Whisper with ROS 2 for humanoid robotics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import PoseStamped

class WhisperROSIntegration(Node):
    """
    ROS 2 integration for Whisper speech recognition
    """
    def __init__(self):
        super().__init__('whisper_ros_integration')

        # Initialize Whisper recognizer
        self.whisper_recognizer = RobustWhisperRecognizer(
            model_size="base",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # ROS 2 publishers
        self.speech_pub = self.create_publisher(String, '/speech_recognition/text', 10)
        self.confidence_pub = self.create_publisher(String, '/speech_recognition/confidence', 10)
        self.activation_pub = self.create_publisher(Bool, '/speech_recognition/activated', 10)

        # ROS 2 subscribers
        self.command_sub = self.create_subscription(
            String, '/speech_recognition/command', self.command_callback, 10
        )

        # Service servers
        self.start_service = self.create_service(
            Trigger, '/speech_recognition/start', self.start_recognition
        )
        self.stop_service = self.create_service(
            Trigger, '/speech_recognition/stop', self.stop_recognition
        )

        # Initialize state
        self.recognition_active = False

        # Add callback to Whisper recognizer
        self.whisper_recognizer.add_speech_callback(self.speech_callback)
        self.whisper_recognizer.add_vad_callback(self.vad_callback)

        self.get_logger().info('Whisper ROS 2 integration initialized')

    def speech_callback(self, speech_result: SpeechResult):
        """Handle speech recognition results"""
        # Publish recognized text
        text_msg = String()
        text_msg.data = speech_result.text
        self.speech_pub.publish(text_msg)

        # Publish confidence
        confidence_msg = String()
        confidence_msg.data = f"{speech_result.confidence:.2f}"
        self.confidence_pub.publish(confidence_msg)

        self.get_logger().info(f'Recognized: {speech_result.text}')

    def vad_callback(self, event_type: str, value: int):
        """Handle voice activity detection events"""
        if event_type == 'speech_detected':
            # Publish activation signal
            activation_msg = Bool()
            activation_msg.data = True
            self.activation_pub.publish(activation_msg)

    def command_callback(self, msg):
        """Handle commands from other nodes"""
        command = msg.data.lower()

        if command == 'start':
            self.start_recognition()
        elif command == 'stop':
            self.stop_recognition()

    def start_recognition(self, request=None, response=None):
        """Start speech recognition"""
        if not self.recognition_active:
            self.whisper_recognizer.start_listening()
            self.recognition_active = True
            self.get_logger().info('Speech recognition started')

        if response:
            response.success = True
            response.message = 'Speech recognition started'
            return response

    def stop_recognition(self, request=None, response=None):
        """Stop speech recognition"""
        if self.recognition_active:
            self.whisper_recognizer.stop_listening()
            self.recognition_active = False
            self.get_logger().info('Speech recognition stopped')

        if response:
            response.success = True
            response.message = 'Speech recognition stopped'
            return response

def main(args=None):
    rclpy.init(args=args)

    whisper_node = WhisperROSIntegration()

    # Start recognition automatically
    whisper_node.start_recognition()

    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_node.whisper_recognizer.stop_listening()
        whisper_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Wake Word Detection

For humanoid robots, implementing wake word detection is important:

```python
import numpy as np
from scipy import signal
import librosa

class WakeWordDetector:
    """
    Wake word detection for activating speech recognition
    """
    def __init__(self, wake_word="hey robot", threshold=0.7):
        self.wake_word = wake_word.lower()
        self.threshold = threshold
        self.activation_phrases = [
            "hey robot",
            "robot",
            "attention",
            "listen"
        ]

        # Pre-compute reference features for wake words
        self.reference_features = self._compute_reference_features()

    def _compute_reference_features(self):
        """Compute reference features for wake words"""
        features = {}
        for phrase in self.activation_phrases:
            # In practice, this would use more sophisticated features
            # like MFCC, spectrograms, or embeddings
            features[phrase] = hash(phrase)  # Simplified for example
        return features

    def detect_wake_word(self, audio_buffer, sample_rate=16000):
        """Detect wake word in audio buffer"""
        # Convert to text using a lightweight ASR model
        # or use keyword spotting techniques

        # For this example, we'll use a simplified approach
        # In practice, use specialized keyword spotting models
        try:
            # Use Whisper for short segments to detect wake word
            audio_float = audio_buffer.astype(np.float32)
            audio_tensor = torch.from_numpy(audio_float).to('cuda' if torch.cuda.is_available() else 'cpu')

            # Transcribe a short segment
            result = whisper.load_model("tiny").transcribe(audio_tensor, language='en')
            text = result['text'].lower()

            # Check for activation phrases
            for phrase in self.activation_phrases:
                if phrase in text:
                    return True, phrase

        except Exception as e:
            print(f"Wake word detection error: {e}")

        return False, None

class WakeWordWhisperRecognizer(RobustWhisperRecognizer):
    """
    Whisper recognizer with wake word activation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize wake word detector
        self.wake_word_detector = WakeWordDetector()
        self.recognition_armed = False

    def _process_audio_stream(self):
        """Process audio stream with wake word activation"""
        audio_buffer = []

        while self.is_listening:
            try:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                audio_buffer.extend(audio_chunk)

                # Keep only recent audio for wake word detection
                max_buffer_size = int(self.sample_rate * 2.0)  # 2 seconds
                if len(audio_buffer) > max_buffer_size:
                    audio_buffer = audio_buffer[-max_buffer_size:]

                # Check for wake word if not actively recognizing
                if not self.speech_detected and len(audio_buffer) > int(self.sample_rate * 0.5):
                    detected, phrase = self.wake_word_detector.detect_wake_word(
                        np.array(audio_buffer[-int(self.sample_rate):])  # Last 1 second
                    )

                    if detected:
                        self.get_logger().info(f"Wake word '{phrase}' detected")
                        self.recognition_armed = True
                        # Start full recognition process
                        self._activate_full_recognition(audio_buffer)

            except Exception as e:
                print(f"Error in wake word audio processing: {e}")
                time.sleep(0.1)

    def _activate_full_recognition(self, initial_buffer):
        """Activate full speech recognition after wake word"""
        # Process the initial buffer that contained the wake word
        self.current_audio_buffer = initial_buffer.copy()
        self.speech_detected = True
```

### Performance Optimization

For real-time humanoid robot applications, optimize Whisper performance:

```python
class OptimizedWhisperRecognizer(RobustWhisperRecognizer):
    """
    Optimized Whisper recognizer for real-time performance
    """
    def __init__(self, *args, **kwargs):
        # Use smaller model for real-time applications
        kwargs['model_size'] = kwargs.get('model_size', 'base')
        super().__init__(*args, **kwargs)

        # Use half precision for faster inference
        self.model = self.model.half()

        # Pre-allocate tensors to avoid allocation overhead
        self._allocate_buffers()

        # Use threading for non-blocking processing
        self.processing_pool = ThreadPoolExecutor(max_workers=2)

    def _allocate_buffers(self):
        """Pre-allocate audio buffers"""
        self.process_buffer = torch.zeros(480000, dtype=torch.float16, device=self.device)  # 30s buffer

    def _process_speech_buffer(self, audio_buffer):
        """Optimized speech buffer processing"""
        # Submit to thread pool for non-blocking processing
        future = self.processing_pool.submit(self._transcribe_in_thread, audio_buffer)
        return future

    def _transcribe_in_thread(self, audio_buffer):
        """Run transcription in separate thread"""
        start_time = time.time()

        try:
            # Convert and move to pre-allocated buffer
            audio_float = np.array(audio_buffer, dtype=np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float).to(self.device)

            # Transcribe with Whisper
            result = self.model.transcribe(audio_tensor, language='en', task='transcribe')

            # Create and return result
            speech_result = SpeechResult(
                text=result['text'].strip(),
                confidence=self._estimate_confidence(result),
                timestamp=time.time(),
                language=result.get('language', 'en'),
                segments=result.get('segments', [])
            )

            return speech_result

        except Exception as e:
            print(f"Error in threaded transcription: {e}")
            return None
```

### Error Handling and Robustness

Implement error handling for production use:

```python
import logging
from functools import wraps

def robust_whisper_operation(func):
    """Decorator for robust Whisper operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            # Return safe default or handle gracefully
            return None
    return wrapper

class RobustWhisperSystem:
    """
    Production-ready Whisper system with error handling
    """
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.max_errors = 10

        # Initialize Whisper with error handling
        self._initialize_whisper()

    @robust_whisper_operation
    def _initialize_whisper(self):
        """Initialize Whisper with error handling"""
        try:
            self.model = whisper.load_model("base")
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to CPU if CUDA fails
            self.model = whisper.load_model("base", device="cpu")
            self.logger.info("Whisper model loaded on CPU as fallback")

    def safe_transcribe(self, audio_buffer):
        """Safely transcribe audio with error handling"""
        try:
            if len(audio_buffer) == 0:
                return None

            audio_tensor = torch.from_numpy(audio_buffer).to(self.model.device)
            result = self.model.transcribe(audio_tensor)

            self.error_count = 0  # Reset error counter on success
            return result

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Transcription error #{self.error_count}: {e}")

            if self.error_count >= self.max_errors:
                self.logger.critical("Too many consecutive errors, restarting...")
                self._initialize_whisper()
                self.error_count = 0

            return None
```

### Summary

In this chapter, we've covered:
- OpenAI Whisper fundamentals and architecture
- Basic and advanced Whisper implementations for robotics
- ROS 2 integration for humanoid robots
- Wake word detection for activation
- Performance optimization techniques
- Error handling and robustness measures

In the next chapter, we'll explore LLM-based task planning, learning how to use large language models to interpret user commands and plan robot actions.