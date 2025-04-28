#
# @lc app=leetcode id=121 lang=python3
#
# [121] Best Time to Buy and Sell Stock
#

# @lc code=start
class Solution:
    # def maxProfit(self, prices: List[int]) -> int:
    #     max_profit = 0
    #     for i in range(0, len(prices)-1):
    #         for j in range(i+1, len(prices)):
    #             profit_que = prices[j] - prices[i]
    #             if profit_que > 0 and profit_que > max_profit:
    #                 max_profit = profit_que
    #             else:
    #                 pass

    #     return max_profit

    def maxProfit(self, prices: List[int]) -> int:
        return sorted(prices[prices.index(min(prices)):])[-1] - min(prices)


# @lc code=end
