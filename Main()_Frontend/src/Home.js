import React, { useState } from "react";
import "./Home.css";
import Navbar from "./Components/Navbar";
import SidebarLeft from "./Area/SidebarLeft";
import RealTimeSidebarLeft from "./Area/RealTimeSidebarLeft";
import Footer from "./Area/Footer";
import MapSwap from "./Area/MapSwap";
import Content from "./Area/Content";
import ProVideoSwap from "./Area/ProVideoSwap";
import ClipSwap from "./Area/ClipSwap";
import UploadSwap from "./Area/UploadSwap";
import MainPageSwap from "./Area/MainPageSwap";
import MyPageSwap from "./Area/MyPageSwap";

function Home() {
  const [selectedPersonData, setSelectedPersonData] = useState([]);
  const [selectedVideoUrl, setSelectedVideoUrl] = useState("");
  const [selectedCamData, setSelectedCamData] = useState([]);
  const [selectedAddress, setSelectedAddress] = useState("");
  const [showClipSwap, setShowClipSwap] = useState(false);
  const [showMapSwap, setShowMapSwap] = useState(false);
  const [showProVideoSwap, setShowProVideoSwap] = useState(false);
  const [showUploadSwap, setShowUploadSwap] = useState(false);
  const [showRealTimeSwap, setShowRealTimeSwap] = useState(false);
  const [videoUrls, setVideoUrls] = useState([]);
  const [showMyPageSwap, setShowMyPageSwap] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(0); // Sidebar의 너비를 상태로 관리

  const handleVideoSelect = (personData, videoUrls) => {
    setSelectedPersonData(personData);
    setSelectedVideoUrl("");
    setSelectedCamData([]);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(true);
    setShowUploadSwap(false);
    setShowRealTimeSwap(false);
    setVideoUrls(videoUrls);
    setShowMyPageSwap(false);
  };

  const handlePersonClick = (filterId) => {
    setShowClipSwap(true);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowUploadSwap(false);
    setShowRealTimeSwap(false);
    setShowMyPageSwap(false);
  };

  const handleMapClipSelect = (camData, address) => {
    setSelectedPersonData([]);
    setSelectedVideoUrl("");
    setSelectedCamData(camData);
    setSelectedAddress(address);
    setShowClipSwap(false);
    setShowMapSwap(true);
    setShowProVideoSwap(false);
    setShowUploadSwap(false);
    setShowRealTimeSwap(false);
    setShowMyPageSwap(false);
  };

  const handleClipSwapClick = (videoUrl) => {
    setSelectedVideoUrl(videoUrl);
    setShowClipSwap(true);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowUploadSwap(false);
    setShowRealTimeSwap(false);
    setShowMyPageSwap(false);
  };

  const handleUploadClick = () => {
    setSelectedPersonData([]);
    setSelectedVideoUrl("");
    setSelectedCamData([]);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowUploadSwap(true);
    setShowRealTimeSwap(false);
    setShowMyPageSwap(false);
  };

  const resetSelection = () => {
    setSelectedPersonData([]);
    setSelectedVideoUrl("");
    setSelectedCamData([]);
    setSelectedAddress(""); // 선택된 주소 상태 초기화
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowUploadSwap(false);
    setShowRealTimeSwap(false);
    setShowMyPageSwap(false);
  };

  const handleMyPageClick = () => {
    setSelectedPersonData([]);
    setSelectedVideoUrl("");
    setSelectedCamData([]);
    setSelectedAddress(""); // 선택된 주소 상태 초기화
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowUploadSwap(false);
    setShowRealTimeSwap(false);
    setShowMyPageSwap(true); // MyPageSwap이 보이도록 설정
  };

  const realtimeSelection = () => {
    setSelectedPersonData([]);
    setSelectedVideoUrl("");
    setSelectedCamData([]);
    setSelectedAddress(""); // 선택된 주소 상태 초기화
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowUploadSwap(false);
    setShowRealTimeSwap(true); // 실시간 분석을 클릭했을 때 showRealTimeSwap를 true로 설정
    setShowMyPageSwap(false);
  };

  const isMainPageSwapActive =
    !showMapSwap &&
    !showProVideoSwap &&
    !showClipSwap &&
    !showUploadSwap &&
    !showRealTimeSwap &&
    !showMyPageSwap; // MyPageSwap이 표시되지 않을 때도 MainPageSwap이 활성화되도록 조건 추가

  return (
    <div>
      <Navbar
        resetSelection={resetSelection}
        onUploadClick={handleUploadClick}
        realtimeSelection={realtimeSelection}
        onMyPageClick={handleMyPageClick} // MyPageSwap을 클릭했을 때 handleMyPageClick 함수가 호출되도록 설정
      />
      <div className="Main">
        {/* isMainPageSwapActive가 true가 아닐 때만 SidebarLeft와 Footer를 렌더링 */}
        {!isMainPageSwapActive &&
          (showRealTimeSwap ? (
            <RealTimeSidebarLeft setSidebarWidth={setSidebarWidth} />
          ) : (
            <SidebarLeft
              onAddressSelect={handleMapClipSelect}
              onVideoSelect={handleVideoSelect}
              onPersonClick={handlePersonClick}
            />
          ))}
        {showMapSwap && (
          <MapSwap
            selectedCamData={selectedCamData}
            selectedAddress={selectedAddress} // 선택된 주소를 MapSwap으로 전달
          />
        )}
        {showProVideoSwap && <ProVideoSwap videoUrls={videoUrls} />}
        {showClipSwap && <ClipSwap />}
        {showUploadSwap && <UploadSwap />}
        {showRealTimeSwap && <Content />}
        {showMyPageSwap && <MyPageSwap />}
        {isMainPageSwapActive && <MainPageSwap />}
      </div>
      {!isMainPageSwapActive &&
        !showMapSwap &&
        !showProVideoSwap &&
        !showClipSwap &&
        !showUploadSwap &&
        !showMyPageSwap && (
          <Footer
            onClipSwapClick={handleClipSwapClick}
            clipInfo={selectedPersonData}
          />
        )}
    </div>
  );
}

export default Home;
