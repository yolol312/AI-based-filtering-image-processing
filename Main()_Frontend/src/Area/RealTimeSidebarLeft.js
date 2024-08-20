import React, { useContext, useState, useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import { ImageLogDataContext } from "../Data/ImageLogDataContext";
import { DataContext } from "../Data/DataContext";
import { RealTimeDataContext } from "../Data/RealTimeDataContext";
import "./RealTimeSidebarLeft.css";
import PrevModal from "../Modal/PrevModal";

const RealTimeSidebarLeft = ({ setSidebarWidth }) => {
  const { data } = useContext(DataContext);
  const { imagelogInfo } = useContext(ImageLogDataContext);
  const { setRealTimeData } = useContext(RealTimeDataContext);
  const [isPrevModalOpen, setPrevModalOpen] = useState(false);
  const userId = data.userId;
  const RealTime_Server_IP = process.env.REACT_APP_REALTIME_Server_IP;
  const sidebarRef = useRef(null);

  // 사이드바의 너비를 계산하여 부모 컴포넌트로 전달
  useEffect(() => {
    if (sidebarRef.current) {
      const width = sidebarRef.current.offsetWidth;
      setSidebarWidth(width); // Sidebar의 너비를 부모 컴포넌트로 전달
    }
  }, []);

  const openPrevModal = async (event) => {
    event.preventDefault();

    try {
      await PrevFrame(); // 비동기 작업이 완료되기를 기다림
      setPrevModalOpen(true); // 그 후에 모달을 염
    } catch (error) {
      console.error("Error while fetching previous frame:", error);
    }
  };

  const PrevFrame = async () => {
    const requestData = {
      user_data: {
        user_id: userId,
      },
    };

    try {
      const response = await fetch(
        `${RealTime_Server_IP}/realtime_search_previous_analysis`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestData),
        }
      );

      const data = await response.json();
      console.log("Previous analysis data fetched successfully:", data);

      setRealTimeData({
        cameras: data.origin_videos,
      });
    } catch (err) {
      console.error("Error fetching previous analysis data: ", err);
    }
  };

  const closePrevModal = () => setPrevModalOpen(false);

  return (
    <Box className="RealTimeSidebarLeft" ref={sidebarRef}>
      <div>
        <h1 className="sidebar-title">영상 등장 인물</h1>
        <button className="upload-button" onClick={openPrevModal}>
          이전 분석 기록 불러오기
        </button>
      </div>
      <div className="info-list">
        {imagelogInfo.map((log, index) => (
          <div key={index} className="log-container">
            {Object.keys(log.person_data).map((personKey) => (
              <div key={personKey} className="info-card">
                <div className="image-placeholder">
                  {log.person_image[personKey] ? (
                    <img
                      src={`data:image/jpeg;base64,${log.person_image[personKey]}`}
                      alt={`Person ${personKey}`}
                    />
                  ) : (
                    "No Image"
                  )}
                </div>
                <div className="info-text">
                  <p>ID: {personKey}</p>
                  <p>나이: {log.person_data[personKey]?.Age || "Unknown"}</p>
                  <p>성별: {log.person_data[personKey]?.Gender || "Unknown"}</p>
                  <p>
                    상의: {log.person_data[personKey]?.Upclothes || "Unknown"}
                  </p>
                  <p>
                    하의: {log.person_data[personKey]?.Downclothes || "Unknown"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
      <PrevModal isOpen={isPrevModalOpen} onClose={closePrevModal} />
    </Box>
  );
};

export default RealTimeSidebarLeft;
