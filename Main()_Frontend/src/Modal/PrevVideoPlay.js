import React, { useContext, useState, useRef } from "react";
import { RealTimeDataContext } from "../Data/RealTimeDataContext";
import { LogRealTimeDataContext } from "../Data/LogRealTimeDataContext";
import "./PrevVideoPlay.css";
import Swal from "sweetalert2";

function PrevVideoPlay({ onBack, onClose }) {
  const { RealTimeData, setRealTimeData } = useContext(RealTimeDataContext);
  const { LogRealTimeInfo, updateLogRealTimeInfo, clearLogRealTimeInfo } =
    useContext(LogRealTimeDataContext);
  const [expandedCameras, setExpandedCameras] = useState({});
  const [videoUrl, setVideoUrl] = useState(null);
  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const videoRef = useRef(null);

  const RealTime_Server_IP = process.env.REACT_APP_REALTIME_Server_IP;

  const extractVideoName = (or_video_name) => {
    const regex = /(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}).*\.mp4$/;
    const match = or_video_name.match(regex);
    return match ? `${match[1]}.mp4` : or_video_name;
  };

  const toggleCamera = (cameraIndex) => {
    setExpandedCameras((prevState) => ({
      ...prevState,
      [cameraIndex]: !prevState[cameraIndex],
    }));
  };

  const handleDeletePrev = async () => {
    if (!selectedVideoId) {
      Swal.fire({
        title: "오류",
        text: "삭제할 영상을 선택해 주세요.",
        icon: "error",
        confirmButtonText: "확인",
      });
      return;
    }

    Swal.fire({
      title: "삭제하시겠습니까?",
      text: "삭제 시 이전 정보가 사라집니다.",
      icon: "warning",
      showCancelButton: true,
      confirmButtonText: "삭제",
      cancelButtonText: "취소",
    }).then(async (result) => {
      if (result.isConfirmed) {
        try {
          const response = await fetch(
            `${RealTime_Server_IP}/realtime_delete_previous_analysis_result`,
            {
              method: "DELETE",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                user_data: {
                  or_video_id: selectedVideoId, // 저장된 영상 ID 사용
                },
              }),
            }
          );

          if (!response.ok) {
            throw new Error("Network response was not ok");
          }

          Swal.fire({
            title: "삭제 완료",
            text: "영상이 성공적으로 삭제되었습니다.",
            icon: "success",
            confirmButtonText: "확인",
          });

          // 상태 초기화
          setVideoUrl(null);
          setSelectedVideoId(null);
          clearLogRealTimeInfo();

          // RealTimeData 업데이트 - 삭제된 비디오 제거
          const updatedCameras = RealTimeData.cameras.map((camera) => ({
            ...camera,
            origin_videos: camera.origin_videos.filter(
              (video) => video.or_video_id !== selectedVideoId
            ),
          }));

          // RealTimeData 업데이트를 위해 setRealTimeData를 호출해야 합니다.
          // 여기서 RealTimeData 업데이트 메소드를 제공하는 함수가 있다고 가정합니다.
          setRealTimeData({
            ...RealTimeData,
            cameras: updatedCameras,
          });
        } catch (error) {
          console.error("Error deleting video:", error);
          Swal.fire({
            title: "오류",
            text: "영상 삭제 중 오류가 발생했습니다.",
            icon: "error",
            confirmButtonText: "확인",
          });
        }
      }
    });
  };

  const handleVideoClick = async (or_video_id) => {
    try {
      clearLogRealTimeInfo(); // 기존 로그 초기화

      const videoUrl = `${RealTime_Server_IP}/realtime_search_previous_analysis_result_video?or_video_id=${or_video_id}`;
      console.log("Video URL:", videoUrl);

      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.removeAttribute("src");
        videoRef.current.load();
      }
      setVideoUrl(videoUrl);
      setSelectedVideoId(or_video_id); // 선택된 영상 ID 저장

      const logUrl = `${RealTime_Server_IP}/realtime_search_previous_analysis_result_logs?or_video_id=${or_video_id}`;
      const response = await fetch(logUrl);
      const logData = await response.json();
      console.log("Log Data:", logData);

      if (logData.logs) {
        const parsedLogs = Object.entries(logData.logs).map(([key, log]) => ({
          Person_ID: key,
          Age: log.Age || "N/A",
          Gender: log.Gender || "N/A",
          UpClothes: log.UpClothes || "N/A",
          DownClothes: log.DownClothes || "N/A",
          Track_ID: log["Track ID"] || "N/A",
        }));

        updateLogRealTimeInfo(parsedLogs);
      } else {
        console.error("Failed to fetch logs or logs are empty");
      }
    } catch (error) {
      console.error("Error fetching video or logs:", error);
    }
  };

  return (
    <div className="prev-video-container">
      <div className="prev-video-sidebar">
        <div className="prev-video-tree">
          <h3>영상 목록</h3>
          <ul>
            {RealTimeData.cameras && RealTimeData.cameras.length > 0 ? (
              RealTimeData.cameras.map((camera, cameraIndex) => (
                <li key={cameraIndex}>
                  <div
                    className="camera-title"
                    onClick={() => toggleCamera(cameraIndex)}
                  >
                    카메라 [ {camera.cam_name} ]
                  </div>
                  {expandedCameras[cameraIndex] && (
                    <ul>
                      {camera.origin_videos &&
                      camera.origin_videos.length > 0 ? (
                        camera.origin_videos.map((video, videoIndex) => (
                          <li
                            key={videoIndex}
                            onClick={() => handleVideoClick(video.or_video_id)}
                            className="video-title"
                          >
                            {extractVideoName(video.or_video_name)}
                          </li>
                        ))
                      ) : (
                        <li>No videos available</li>
                      )}
                    </ul>
                  )}
                </li>
              ))
            ) : (
              <li>No cameras available</li>
            )}
          </ul>
        </div>
        <div className="prev-video-records">
          <h3>영상 등장 인물</h3>
          {LogRealTimeInfo.length > 0 ? (
            <div className="log-table">
              {LogRealTimeInfo.map((log, index) => (
                <div key={index} className="log-row">
                  <div className="log-cell">
                    <strong>Person_ID:</strong> {log.Person_ID}
                  </div>
                  <div className="log-cell">
                    <strong>나이:</strong> {log.Age}
                  </div>
                  <div className="log-cell">
                    <strong>성별:</strong> {log.Gender}
                  </div>
                  <div className="log-cell">
                    <strong>상의:</strong> {log.UpClothes}
                  </div>
                  <div className="log-cell">
                    <strong>하의:</strong> {log.DownClothes}
                  </div>
                  <div className="log-cell">
                    <strong>Track ID:</strong> {log.Track_ID}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p>No logs available</p>
          )}
        </div>
      </div>
      <div className="prev-video-main">
        <video ref={videoRef} controls className="prev-video-player" autoPlay>
          {videoUrl && <source src={videoUrl} type="video/mp4" />}
        </video>
        <button className="delete-button" onClick={handleDeletePrev}>
          영상 삭제
        </button>
      </div>
    </div>
  );
}

export default PrevVideoPlay;
