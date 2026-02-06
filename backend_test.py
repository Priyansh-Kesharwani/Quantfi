#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime, timedelta
import time

class FinancialPlatformTester:
    def __init__(self, base_url="https://smartdca.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if details:
            print(f"    {details}")
        
        if success:
            self.tests_passed += 1
        else:
            self.failed_tests.append(f"{name}: {details}")
        print()

    def run_test(self, name, method, endpoint, expected_status=200, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, timeout=timeout)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=timeout)
            elif method == 'PUT':
                response = self.session.put(url, json=data, timeout=timeout)
            elif method == 'DELETE':
                response = self.session.delete(url, timeout=timeout)
            
            success = response.status_code == expected_status
            
            if success:
                try:
                    response_data = response.json()
                    self.log_test(name, True, f"Status: {response.status_code}")
                    return True, response_data
                except:
                    self.log_test(name, True, f"Status: {response.status_code} (No JSON response)")
                    return True, {}
            else:
                try:
                    error_data = response.json()
                    self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}. Error: {error_data}")
                except:
                    self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}. Response: {response.text[:200]}")
                return False, {}

        except requests.exceptions.Timeout:
            self.log_test(name, False, f"Request timed out after {timeout}s")
            return False, {}
        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        print("🔍 Testing Health Check...")
        success, data = self.run_test("Health Check", "GET", "health")
        
        if success and data.get('status') == 'healthy':
            self.log_test("Health Check Content", True, "Status is healthy")
        elif success:
            self.log_test("Health Check Content", False, f"Unexpected status: {data.get('status')}")
        
        return success

    def test_add_gold_asset(self):
        """Test adding GOLD asset"""
        print("🔍 Testing Add GOLD Asset...")
        
        asset_data = {
            "symbol": "GOLD",
            "name": "Gold Futures",
            "asset_type": "metal"
        }
        
        success, data = self.run_test("Add GOLD Asset", "POST", "assets", 200, asset_data)
        
        if success:
            if data.get('symbol') == 'GOLD' and data.get('asset_type') == 'metal':
                self.log_test("GOLD Asset Data", True, f"Symbol: {data.get('symbol')}, Type: {data.get('asset_type')}")
            else:
                self.log_test("GOLD Asset Data", False, f"Unexpected data: {data}")
        
        return success

    def test_get_assets(self):
        """Test getting assets list"""
        print("🔍 Testing Get Assets...")
        success, data = self.run_test("Get Assets", "GET", "assets")
        
        if success:
            if isinstance(data, list):
                gold_found = any(asset.get('symbol') == 'GOLD' for asset in data)
                if gold_found:
                    self.log_test("GOLD in Assets List", True, f"Found {len(data)} assets including GOLD")
                else:
                    self.log_test("GOLD in Assets List", False, f"GOLD not found in {len(data)} assets")
            else:
                self.log_test("Assets List Format", False, f"Expected list, got: {type(data)}")
        
        return success

    def test_gold_price_fetching(self):
        """Test fetching GOLD price data"""
        print("🔍 Testing GOLD Price Fetching...")
        
        # Test latest price
        success, data = self.run_test("Get GOLD Latest Price", "GET", "prices/GOLD", timeout=45)
        
        if success:
            required_fields = ['symbol', 'price_usd', 'price_inr', 'usd_inr_rate']
            missing_fields = [field for field in required_fields if field not in data]
            
            if not missing_fields:
                self.log_test("GOLD Price Data Fields", True, 
                            f"USD: ${data.get('price_usd'):.2f}, INR: ₹{data.get('price_inr'):.2f}")
            else:
                self.log_test("GOLD Price Data Fields", False, f"Missing fields: {missing_fields}")
        
        # Test historical price data
        success2, hist_data = self.run_test("Get GOLD Historical Prices", "GET", "prices/GOLD/history?period=1y", timeout=45)
        
        if success2:
            if 'history' in hist_data and isinstance(hist_data['history'], list):
                history_len = len(hist_data['history'])
                if history_len > 0:
                    self.log_test("GOLD Historical Data", True, f"Retrieved {history_len} historical data points")
                else:
                    self.log_test("GOLD Historical Data", False, "No historical data returned")
            else:
                self.log_test("GOLD Historical Data", False, f"Invalid format: {type(hist_data.get('history'))}")
        
        return success and success2

    def test_technical_indicators(self):
        """Test technical indicators calculation"""
        print("🔍 Testing Technical Indicators...")
        
        success, data = self.run_test("Get GOLD Technical Indicators", "GET", "indicators/GOLD", timeout=60)
        
        if success:
            # Check for key indicators
            key_indicators = ['sma_50', 'sma_200', 'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr_14', 'adx_14']
            present_indicators = [ind for ind in key_indicators if data.get(ind) is not None]
            
            if len(present_indicators) >= 6:  # At least 6 out of 8 indicators
                self.log_test("Technical Indicators Coverage", True, 
                            f"Found {len(present_indicators)}/{len(key_indicators)} indicators")
                
                # Check specific values are reasonable
                rsi = data.get('rsi_14')
                if rsi and 0 <= rsi <= 100:
                    self.log_test("RSI Value Range", True, f"RSI: {rsi:.2f}")
                elif rsi:
                    self.log_test("RSI Value Range", False, f"RSI out of range: {rsi}")
                
            else:
                self.log_test("Technical Indicators Coverage", False, 
                            f"Only {len(present_indicators)}/{len(key_indicators)} indicators present")
        
        return success

    def test_dca_scoring(self):
        """Test DCA scoring with LLM explanation"""
        print("🔍 Testing DCA Scoring...")
        
        success, data = self.run_test("Get GOLD DCA Score", "GET", "scores/GOLD", timeout=90)
        
        if success:
            # Check score structure
            required_fields = ['composite_score', 'zone', 'breakdown', 'explanation']
            missing_fields = [field for field in required_fields if field not in data]
            
            if not missing_fields:
                score = data.get('composite_score')
                zone = data.get('zone')
                explanation = data.get('explanation', '')
                
                # Validate score range
                if 0 <= score <= 100:
                    self.log_test("DCA Score Range", True, f"Score: {score:.1f}/100, Zone: {zone}")
                else:
                    self.log_test("DCA Score Range", False, f"Score out of range: {score}")
                
                # Check explanation length (should be substantial if LLM is working)
                if len(explanation) > 50:
                    self.log_test("LLM Explanation", True, f"Generated {len(explanation)} character explanation")
                else:
                    self.log_test("LLM Explanation", False, f"Short explanation ({len(explanation)} chars): {explanation}")
                
                # Check breakdown structure
                breakdown = data.get('breakdown', {})
                breakdown_fields = ['technical_momentum', 'volatility_opportunity', 'statistical_deviation', 'macro_fx']
                if all(field in breakdown for field in breakdown_fields):
                    self.log_test("Score Breakdown", True, "All breakdown components present")
                else:
                    missing_breakdown = [field for field in breakdown_fields if field not in breakdown]
                    self.log_test("Score Breakdown", False, f"Missing breakdown fields: {missing_breakdown}")
                
            else:
                self.log_test("DCA Score Structure", False, f"Missing fields: {missing_fields}")
        
        return success

    def test_settings_api(self):
        """Test settings GET and PUT endpoints"""
        print("🔍 Testing Settings API...")
        
        # Test GET settings
        success1, settings_data = self.run_test("Get Settings", "GET", "settings")
        
        if success1:
            expected_fields = ['default_dca_cadence', 'default_dca_amount', 'score_weights']
            missing_fields = [field for field in expected_fields if field not in settings_data]
            
            if not missing_fields:
                self.log_test("Settings Structure", True, f"DCA Amount: {settings_data.get('default_dca_amount')}")
            else:
                self.log_test("Settings Structure", False, f"Missing fields: {missing_fields}")
        
        # Test PUT settings (update)
        if success1:
            updated_settings = settings_data.copy()
            updated_settings['default_dca_amount'] = 6000  # Change from default 5000
            
            success2, _ = self.run_test("Update Settings", "PUT", "settings", 200, updated_settings)
            
            if success2:
                # Verify the update
                success3, verify_data = self.run_test("Verify Settings Update", "GET", "settings")
                if success3 and verify_data.get('default_dca_amount') == 6000:
                    self.log_test("Settings Update Verification", True, "DCA amount updated successfully")
                else:
                    self.log_test("Settings Update Verification", False, 
                                f"Expected 6000, got {verify_data.get('default_dca_amount')}")
            
            return success1 and success2
        
        return success1

    def test_dashboard_endpoint(self):
        """Test dashboard overview endpoint"""
        print("🔍 Testing Dashboard Endpoint...")
        
        success, data = self.run_test("Get Dashboard", "GET", "dashboard", timeout=60)
        
        if success:
            if 'assets' in data and isinstance(data['assets'], list):
                assets_count = len(data['assets'])
                self.log_test("Dashboard Structure", True, f"Dashboard contains {assets_count} assets")
                
                # Check if GOLD asset has complete data
                gold_asset = next((asset for asset in data['assets'] if asset.get('asset', {}).get('symbol') == 'GOLD'), None)
                if gold_asset:
                    has_price = gold_asset.get('price') is not None
                    has_score = gold_asset.get('score') is not None
                    has_indicators = gold_asset.get('indicators') is not None
                    
                    complete_data = has_price and has_score and has_indicators
                    self.log_test("GOLD Dashboard Data", complete_data, 
                                f"Price: {has_price}, Score: {has_score}, Indicators: {has_indicators}")
                else:
                    self.log_test("GOLD in Dashboard", False, "GOLD asset not found in dashboard")
            else:
                self.log_test("Dashboard Structure", False, f"Invalid dashboard format: {type(data.get('assets'))}")
        
        return success

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Financial Platform Backend Tests")
        print(f"🎯 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Run tests in logical order
        tests = [
            self.test_health_check,
            self.test_add_gold_asset,
            self.test_get_assets,
            self.test_gold_price_fetching,
            self.test_technical_indicators,
            self.test_dca_scoring,
            self.test_settings_api,
            self.test_dashboard_endpoint
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                self.log_test(f"{test.__name__}", False, f"Test execution error: {str(e)}")
        
        # Print summary
        print("=" * 60)
        print(f"📊 TEST SUMMARY")
        print(f"✅ Passed: {self.tests_passed}/{self.tests_run}")
        print(f"❌ Failed: {len(self.failed_tests)}")
        
        if self.failed_tests:
            print("\n🔍 FAILED TESTS:")
            for failure in self.failed_tests:
                print(f"  • {failure}")
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        return self.tests_passed == self.tests_run

def main():
    tester = FinancialPlatformTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())