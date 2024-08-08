import React, { useState, useContext, useEffect, useRef } from 'react';
import './FileUploadFiltering.css';
import Swal from 'sweetalert2';
import { DataContext } from '../Data/DataContext';
import { ProgressBarDataContext } from '../Data/ProgressBarDataContext';

const Server_IP = process.env.REACT_APP_Server_IP;

function FileUploadFiltering() {
  const { data, setData } = useContext(DataContext); // DataContext에서 setData 가져오기
  const { updateProgressBarInfo, ProgressBarInfo, setProgressBarInfo } = useContext(ProgressBarDataContext);
  const [videoFile, setVideoFile] = useState(null);
  const [address, setAddress] = useState('');
  const [camName, setCamName] = useState('');
  const [year, setYear] = useState('2020');
  const [month, setMonth] = useState('01');
  const [day, setDay] = useState('01');
  const [hour, setHour] = useState('00');
  const [minute, setMinute] = useState('00');
  const [second, setSecond] = useState('00');
  const [startTime, setStartTime] = useState('');
  const [bundleName, setbundleName] = useState(''); // 구역 이름 상태 추가
  const [list, setList] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [age, setAge] = useState('');
  const [color, setColor] = useState('');
  const [clothes, setClothes] = useState('');
  const [gender, setGender] = useState('');
  const [createClip, setCreateClip] = useState(true);
  const [addressOptions, setAddressOptions] = useState([]);
  const [filteredCamOptions, setFilteredCamOptions] = useState([]);
  const completeRef = useRef(null);
  const intervalRef = useRef(null);
  const [width, setWidth] = useState(0); // width 상태 추가

  const userId = data.userId;

  useEffect(() => {
    if (data.mapcameraprovideoinfo && data.mapcameraprovideoinfo.length > 0) {
      const uniqueAddresses = [...new Set(data.mapcameraprovideoinfo.map((cam) => cam.address))];
      setAddressOptions(uniqueAddresses);
    }
  }, [data.mapcameraprovideoinfo]);

  useEffect(() => {
    if (address) {
      const cams = data.mapcameraprovideoinfo.filter((cam) => cam.address === address);
      setFilteredCamOptions(cams);
      setCamName(''); // 주소 변경 시 캠 이름 초기화
    } else {
      setFilteredCamOptions([]);
    }
  }, [address, data.mapcameraprovideoinfo]);

  useEffect(() => {
    setStartTime(formatDateTime(year, month, day, hour, minute, second));
  }, [year, month, day, hour, minute, second]);

  const formatDateTime = (year, month, day, hour, minute, second) => {
    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
  };

  const handleAddToList = async () => {
    if (!videoFile || !address || !camName || !startTime) {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '모든 필드를 입력하세요.',
      });
      return;
    }

    if (videoFile.type !== 'video/mp4') {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '오직 MP4 파일만 업로드 가능합니다.',
      });
      return;
    }

    try {
      const videoContent = await toBase64(videoFile);

      const requestData = {
        user_id: userId,
        video_data: [
          {
            video_name: videoFile.name,
            video_content: videoContent,
          },
        ],
      };

      const response = await fetch(`${Server_IP}/upload_progresstime`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (response.ok) {
        const responseData = await response.json();
        const { total_length } = responseData;
      
        // total_length 값을 ProgressBarDataContext에 업데이트
        updateProgressBarInfo(total_length);
      
        Swal.fire({
          icon: 'success',
          title: '성공',
          text: `비디오가 성공적으로 업로드되었습니다.` //총 길이: ${responseData.total_length}초
        });

        setList([
          ...list,
          { file: videoFile, fileName: videoFile.name, address, camName, startTime },
        ]);
        setVideoFile(null);
        setAddress('');
        setCamName('');
        setStartTime('');
        document.getElementById('videoFileInput').value = '';
      } else {
        console.error('서버 응답 에러:', response.statusText);
        const responseData = await response.json();
        Swal.fire({
          icon: 'error',
          title: '실패',
          text: responseData.error || '비디오 업로드에 실패했습니다.', // 서버에서 반환된 오류 메시지를 사용
        });
      }
    } catch (error) {
      console.error('비디오 업로드 중 오류 발생:', error);
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '비디오 업로드 중 오류가 발생했습니다.',
      });
    }
  };

  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
  };

  const handleImageFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
    } else {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '이미지 파일만 업로드 가능합니다.',
      });
      event.target.value = ''; // 파일 입력 필드 초기화
    }
  };

  const startProgress = (duration) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    if (completeRef.current) {
      clearInterval(completeRef.current);
    }

    setWidth(0); // 초기화 후 일정 시간 대기
    const intervalTime = (duration * 1000) / 100; // 100%를 달성하기 위한 시간 간격 계산

    setTimeout(() => {
      intervalRef.current = setInterval(() => {
        setWidth((prevWidth) => {
          if (prevWidth >= 97) {
            clearInterval(intervalRef.current);
            return 97;
          }
          return prevWidth + 1;
        });
      }, intervalTime); // 동적으로 계산된 시간 간격
    }, 100); // 100ms 대기 후 시작
  };

  const completeProgress = () => {
    if (completeRef.current) {
      clearInterval(completeRef.current);
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    completeRef.current = setInterval(() => {
      setWidth((prevWidth) => {
        if (prevWidth >= 100) {
          clearInterval(completeRef.current);
          return 100;
        }
        return prevWidth + 1;
      });
    }, 20); // 20ms 간격으로 애니메이션 효과
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (completeRef.current) {
        clearInterval(completeRef.current);
      }
    };
  }, []);

  const handleSubmit = async () => {
    if (list.length === 0) {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '최소 하나의 비디오 파일을 리스트에 추가하세요.',
      });
      return;
    }

    if (!age && !color && !clothes && !gender) {
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '최소 하나의 필터 조건을 설정하세요.',
      });
      return;
    }

    try {
      const videoData = await Promise.all(
        list.map(async (item) => ({
          video_name: item.fileName,
          video_content: await toBase64(item.file),
          address: item.address,
          cam_name: item.camName,
          start_time: item.startTime,
        })),
      );

      let imageData = {};
      if (selectedFile) {
        imageData = {
          image_name: selectedFile.name,
          image_content: await toBase64(selectedFile),
        };
      }

      const user_data = {
        user_id: userId,
      };

      const bundle_data = {
        bundle_name: bundleName,
      };

      const filter_data = {
        age: age || null,
        color: color || null,
        type: clothes || null,
        gender: gender || null,
      };

      const flag_data = {
        clip_flag: createClip ? 'true' : 'false',
      };

      const data = {
        video_data: videoData,
        user_data: user_data,
        filter_data: filter_data,
        flag_data: flag_data,
        bundle_data: bundle_data,
      };

      if (selectedFile) {
        data.image_data = imageData;
      }

      console.log('보낼 JSON:', JSON.stringify(data));
      console.log('bundleName:', bundleName); // 추가된 로그

      startProgress(ProgressBarInfo);

      const response = await fetch(`${Server_IP}/upload_file`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const responseData = await response.json(); // 응답 데이터 파싱
      if (response.ok) {
        setData((prevData) => ({
          ...prevData,
          filterid: responseData.filter_id,
        })); // filter_id를 DataContext에 저장
        Swal.fire({
          icon: 'success',
          title: '성공',
          text: '파일이 성공적으로 업로드되었습니다.',
        });
        completeProgress();
      } else {
        Swal.fire({
          icon: 'error',
          title: '실패',
          text: '파일 업로드에 실패했습니다.',
        });
      }
    } catch (error) {
      console.error('파일 업로드 중 오류 발생:', error);
      Swal.fire({
        icon: 'error',
        title: '오류',
        text: '파일 업로드 중 오류가 발생했습니다.',
      });
    }
  };

  const toBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = (error) => reject(error);
    });
  };

  return (
    <div className="file-upload-filtering">
      <div className="file-upload">
        <h2>파일 업로드</h2>
        <select value={address} onChange={(e) => setAddress(e.target.value)}>
          <option value="">주소 선택</option>
          {addressOptions.map((addr, index) => (
            <option key={index} value={addr}>
              {addr}
            </option>
          ))}
        </select>
        <select value={camName} onChange={(e) => setCamName(e.target.value)} disabled={!address}>
          <option value="">캠 이름 선택</option>
          {filteredCamOptions.map((cam, index) => (
            <option key={index} value={cam.cam_name}>
              {cam.cam_name}
            </option>
          ))}
        </select>
        <div className="date-time-picker">
          <select value={year} onChange={(e) => setYear(e.target.value)}>
            {[...Array(100)].map((_, i) => (
              <option key={i} value={2000 + i}>
                {2000 + i}
              </option>
            ))}
          </select>
          <label>년</label>
          <select value={month} onChange={(e) => setMonth(e.target.value)}>
            {[...Array(12)].map((_, i) => (
              <option key={i} value={String(i + 1).padStart(2, '0')}>
                {String(i + 1).padStart(2, '0')}
              </option>
            ))}
          </select>
          <label>월</label>
          <select value={day} onChange={(e) => setDay(e.target.value)}>
            {[...Array(31)].map((_, i) => (
              <option key={i} value={String(i + 1).padStart(2, '0')}>
                {String(i + 1).padStart(2, '0')}
              </option>
            ))}
          </select>
          <label>일</label>
          <select value={hour} onChange={(e) => setHour(e.target.value)}>
            {[...Array(24)].map((_, i) => (
              <option key={i} value={String(i).padStart(2, '0')}>
                {String(i).padStart(2, '0')}
              </option>
            ))}
          </select>
          <label>시</label>
          <select value={minute} onChange={(e) => setMinute(e.target.value)}>
            {[...Array(60)].map((_, i) => (
              <option key={i} value={String(i).padStart(2, '0')}>
                {String(i).padStart(2, '0')}
              </option>
            ))}
          </select>
          <label>분</label>
          <select value={second} onChange={(e) => setSecond(e.target.value)}>
            {[...Array(60)].map((_, i) => (
              <option key={i} value={String(i).padStart(2, '0')}>
                {String(i).padStart(2, '0')}
              </option>
            ))}
          </select>
          <label>초</label>
        </div>
        <div className="form-group">
          <label>구역 이름</label> {/* 구역 이름 레이블 추가 */}
          <input
            type="text"
            value={bundleName}
            onChange={(e) => setbundleName(e.target.value)}
            placeholder="구역 이름 입력"
          />
        </div>
        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={createClip}
              onChange={(e) => setCreateClip(e.target.checked)}
            />
            {createClip ? ' 클립 생성' : ' 클립 미생성'}
          </label>
        </div>
        <input type="file" id="videoFileInput" accept="video/mp4" onChange={handleFileChange} />
        <div className="buttons">
          <button onClick={handleAddToList} className="add-to-list-button">
            리스트 추가
          </button>
        </div>
        <ul className="file-list">
          {list.map((item, index) => (
            <li key={index}>
              <span className="file-name">{item.fileName}</span>
              <span className="address">{item.address}</span>
              <span className="cam-name">{item.camName}</span>
              <span className="start-time">{item.startTime}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="filtering">
        <h2>필터링</h2>
        <div className="form-group">
          <label>이미지 파일:</label>
          <input type="file" accept="image/*" onChange={handleImageFileChange} />
        </div>
        <div className="form-group">
          <label>나이:</label>
          <select value={age} onChange={(e) => setAge(e.target.value)}>
            <option value="">선택하지 않음</option>
            <option value="Child">유아</option>
            <option value="Youth">청년</option>
            <option value="Middle">중년</option>
            <option value="Old">노년</option>
          </select>
        </div>
        <div className="form-group">
          <label>옷 색상:</label>
          <select value={color} onChange={(e) => setColor(e.target.value)}>
            <option value="">선택하지 않음</option>
            <option value="black">검정색</option>
            <option value="white">하얀색</option>
            <option value="red">빨간색</option>
            <option value="yellow">노란색</option>
            <option value="green">초록색</option>
            <option value="blue">파란색</option>
            <option value="grey">갈색</option>
          </select>
        </div>
        <div className="form-group">
          <label>상하의 종류:</label>
          <select value={clothes} onChange={(e) => setClothes(e.target.value)}>
            <option value="">선택하지 않음</option>
            <option value="dress">드레스</option>
            <option value="longsleevetop">긴팔상의</option>
            <option value="shortsleevetop">반팔상의</option>
            <option value="vest">조끼</option>
            <option value="shorts">반바지</option>
            <option value="pants">긴바지</option>
            <option value="skirt">치마</option>
          </select>
        </div>
        <div className="form-group">
          <label>성별:</label>
          <div className="radio-group">
            <label>
              <input
                type="radio"
                value="남성"
                checked={gender === '남성'}
                onChange={(e) => setGender(e.target.value)}
              />
              남성
            </label>
            <label>
              <input
                type="radio"
                value="여성"
                checked={gender === '여성'}
                onChange={(e) => setGender(e.target.value)}
              />
              여성
            </label>
          </div>
        </div>
        <button onClick={handleSubmit} className="submit-button">
          제출하기
        </button>
        <div id="progress-container">
        <div id="progress-bar" style={{ width: `${width}%` }}>
          {width}%
        </div>
      </div>
      </div>
    </div>
  );
}

export default FileUploadFiltering;
