import React from "react";
import "./MainPageSwap.css";

function MainPageSwap() {
  return (
    <div className="MainScreen">
      <h1 className="main-title">AI기반 특정 인물 탐색 및 추적</h1>
      <div className="content">
        <h3 className="section-title">실시간 분석</h3>
        <p className="section-paragraph">
          카메라에 현재 찍히고 있는 영상을 실시간으로 분석합니다.
        </p>
        <h3 className="section-title">영상 분석</h3>
        <p className="section-paragraph">
          분석할 지도 영역을 등록하고 필터링 선택과 함께 영상을 업로드하면
          조건에 맞게 분석합니다.
        </p>
      </div>
    </div>
  );
}

export default MainPageSwap;
