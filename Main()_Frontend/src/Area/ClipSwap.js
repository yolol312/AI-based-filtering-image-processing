import React, { useContext, useRef, useState, useEffect } from "react";
import { DataContext } from "../Data/DataContext";
import { PersonDataContext } from "../Data/PersonDataContext";
import { ClipDataContext } from "../Data/ClipDataContext";
import { MapDataContext } from "../Data/MapDataContext"; // MapDataContext 추가
import RouteModal from "../Modal/RouteModal";
import "./ClipSwap.css";
import RouteDraw from "../Modal/RouteDraw";

const Server_IP = process.env.REACT_APP_Server_IP;

function ClipSwap() {
  const videoRef = useRef();
  const { data } = useContext(DataContext);
  const { personData = [] } = useContext(PersonDataContext);
  const { clipInfo, updateClipInfo } = useContext(ClipDataContext);
  const { updateMapInfo } = useContext(MapDataContext); // MapDataContext에서 updateMapInfo 가져오기
  const [isRouteModalOpen, setRouteModalOpen] = useState(false);
  const [showClips, setShowClips] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);

  const userId = data?.userId;

  useEffect(() => {
    if (videoRef.current && videoUrl) {
      videoRef.current.load(); // 새로운 비디오 URL이 설정될 때 비디오를 다시 로드
      videoRef.current.play(); // 자동 재생
    }
  }, [videoUrl]);

  const openRouteModal = (event) => {
    event.preventDefault();
    setRouteModalOpen(true);
  };

  const closeRouteModal = () => setRouteModalOpen(false);

  const handlePersonClick = async (personID, filterID) => {
    try {
      const response = await fetch(
        `${Server_IP}/get_Person_to_clip?user_id=${userId}&person_id=${personID}&filter_id=${filterID}`
      );
      if (response.ok) {
        const result = await response.json();
        updateClipInfo(result.clip_info); // ClipDataContext에 clip_info 저장
        setShowClips(true); // 클립 보기 버튼 클릭 시 클립 표시
        console.log("clip data saved:", result);
      } else {
        console.error("Failed to fetch person data");
      }
    } catch (error) {
      console.error("Error fetching person data:", error);
    }
  };

  const handlePathClick = async (personId) => {
    try {
      const response = await fetch(
        `${Server_IP}/map_cal?person_id=${personId}`
      );
      const responseData = await response.json();
      if (response.ok) {
        updateMapInfo(responseData);
        console.log("Map data saved:", responseData);
      } else {
        console.error("Failed to fetch map data:", responseData);
      }
    } catch (error) {
      console.error("Error fetching map data:", error);
    }
  };

  const handleVideoClick = (video) => {
    const videoUrl = `${Server_IP}/stream_clipvideo?clip_video=${encodeURIComponent(
      video
    )}`;
    console.log("Video URL:", videoUrl);
    setVideoUrl(videoUrl); // 비디오 URL을 상태에 저장하여 스트리밍
  };

  return (
    <div className="ClipSwap">
      <div className="left-panel">
        <div className="person-info">
          <h3>Person 정보</h3>
          <div className="info-container">
            <div className="person-image">
              {personData.length > 0 && personData[0].person_origin_face ? (
                <img
                  src={`data:image/jpeg;base64,${personData[0].person_origin_face}`}
                  alt={`Person`}
                />
              ) : (
                <div className="empty-image">이미지 없음</div>
              )}
            </div>
            <div className="person-details">
              <p>
                나이: {personData.length > 0 ? personData[0].person_age : ""}
              </p>
              <p>
                상의:{" "}
                {personData.length > 0 ? personData[0].person_upclothes : ""}
              </p>
              <p>
                하의:{" "}
                {personData.length > 0 ? personData[0].person_downclothes : ""}
              </p>
              <p>
                성별: {personData.length > 0 ? personData[0].person_gender : ""}
              </p>
            </div>
          </div>
          <div className="buttons">
            <button
              className="clip-button"
              onClick={() =>
                handlePersonClick(
                  personData[0].person_id,
                  personData[0].filter_id
                )
              }
            >
              클립 보기
            </button>
            <button
              className="route-button"
              onClick={() => handlePathClick(personData[0].person_id)}
            >
              경로 보기
            </button>
          </div>
        </div>
        <div className="map-view">
          <button
            className="zoom-button"
            onClick={(event) => {
              openRouteModal(event); // event 객체를 전달하여 모달을 열고
              handlePathClick(personData[0].person_id); // 경로보기 이벤트 실행
            }}
          ></button>
          <RouteDraw />
        </div>
      </div>
      <div className="right-panel">
        <div className="clip-video-wrapper">
          <video ref={videoRef} controls className="clip-video" autoPlay>
            {videoUrl && <source src={videoUrl} type="video/mp4" />}
          </video>
        </div>
        <div className="clip-list">
          {showClips &&
            clipInfo &&
            clipInfo
              .sort((a, b) => new Date(a.clip_time) - new Date(b.clip_time)) // 시간 순서대로 정렬
              .map((clip, index) => (
                <div
                  key={index}
                  className="clip-item"
                  onClick={() => handleVideoClick(clip.clip_video)}
                >
                  <p>{clip.cam_name}</p>
                  <p>{clip.clip_time}</p>
                </div>
              ))}
          {!showClips}
        </div>
      </div>
      {/* RouteModal 열기 */}
      <RouteModal isOpen={isRouteModalOpen} onClose={closeRouteModal} />
    </div>
  );
}

export default ClipSwap;
