import React, { useRef, useState } from "react";
import ReactPlayer from "react-player";
import "./ProVideoSwap.css";

const ProVideoSwap = ({ videoUrls = [] }) => {
  const playerRefs = [useRef(null), useRef(null), useRef(null), useRef(null)];
  const [playing, setPlaying] = useState(false);
  const [played, setPlayed] = useState(0);
  const [selectedVideo, setSelectedVideo] = useState(null);

  const togglePlayPause = () => {
    setPlaying(!playing);
  };

  const handleProgress = (state) => {
    if (playing) {
      setPlayed(state.played);
    }
  };

  const handleSeek = (e) => {
    const newTime = parseFloat(e.target.value);
    setPlayed(newTime);
    playerRefs.forEach((ref) => {
      if (ref.current) {
        ref.current.seekTo(newTime);
      }
    });
  };

  const handleVideoClick = (index) => {
    setSelectedVideo(index);
  };

  const handleBackToGrid = () => {
    setSelectedVideo(null);
  };

  return (
    <div>
      {selectedVideo === null ? (
        <div className="video-grid">
          {videoUrls.slice(0, 4).map((url, index) => (
            <div key={index} onClick={() => handleVideoClick(index)}>
              <ReactPlayer
                ref={playerRefs[index]}
                url={url}
                playing={playing}
                controls={false}
                width="100%"
                height="100%"
                onProgress={handleProgress}
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="video-fullscreen" onClick={handleBackToGrid}>
          <ReactPlayer
            ref={playerRefs[selectedVideo]}
            url={videoUrls[selectedVideo]}
            playing={playing}
            controls={true}
            width="100%"
            height="100%"
            onProgress={handleProgress}
          />
        </div>
      )}
      <div className="controls">
        <div
          className="thumbnail-bar"
          style={{
            backgroundImage: "url(/combined_thumbnail.jpg)",
            backgroundSize: "cover",
          }}
        >
          <input
            type="range"
            min={0}
            max={1}
            step="any"
            value={played}
            onChange={handleSeek}
            className="seek-bar"
          />
        </div>
        <button onClick={togglePlayPause} className="control-button">
          {playing ? "||" : "▶"}
        </button>
      </div>
    </div>
  );
};

export default ProVideoSwap;
